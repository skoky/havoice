//! Internet radio on the Voice PE.
//!
//!     cargo run --example radio                 # preset stations
//!     cargo run --example radio -- jazz         # search radio-browser.info (MP3 streams)
//!
//! On the device: double press = next station, triple press = previous,
//! long press = stop and quit; the dial changes volume as usual. Ctrl-C also stops.
//!
//! The device fetches the stream itself (HTTP or HTTPS) - the Mac only tells it which URL.
//! The firmware decodes MP3 and FLAC, so AAC / HLS / Ogg stations won't play.
//! "Now playing" comes from the stream's ICY metadata, read here on the Mac.

use std::time::Duration;

use anyhow::{Result, bail};
use havoice::{Event, MediaPlayerCommand, Rgb, VoicePe};
use tokio::task::JoinHandle;

const PRESETS: &[(&str, &str)] = &[
    (
        "ČRo Radiožurnál",
        "https://rozhlas.stream/radiozurnal_mp3_128.mp3",
    ),
    ("ČRo Jazz", "https://rozhlas.stream/jazz.mp3"),
    ("ČRo Radio Wave", "http://rozhlas.stream/wave_mp3_256.mp3"),
    (
        "Radio Paradise",
        "http://stream-uk1.radioparadise.com/mp3-128",
    ),
    (
        "Radio Paradise Mellow",
        "http://stream.radioparadise.com/mellow-192",
    ),
];

/// radio-browser.info asks clients to identify themselves.
const USER_AGENT: &str = "havoice-example/0.1";

struct Station {
    name: String,
    url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let query: Vec<String> = std::env::args().skip(1).collect();
    let stations = if query.is_empty() {
        PRESETS
            .iter()
            .map(|(n, u)| Station {
                name: n.to_string(),
                url: u.to_string(),
            })
            .collect()
    } else {
        search(&query.join(" ")).await?
    };
    for (i, s) in stations.iter().enumerate() {
        println!("{:2}. {}", i + 1, s.name);
    }

    let pe = VoicePe::from_env().await?;
    let mut events = pe.events().await?;
    println!("\ndouble press = next, triple = previous, long press = stop (or Ctrl-C)\n");

    let mut i = 0;
    let mut titles = tune(&pe, &stations, i).await?;
    let mut volume = None;
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        let ev = tokio::select! {
            _ = &mut ctrl_c => break,
            ev = events.next() => ev,
        };
        let step = match ev {
            None => {
                eprintln!("connection closed: {}", pe.client().close_reason());
                break;
            }
            Some(Event::Button(press)) => match press.as_str() {
                "double_press" => 1,
                "triple_press" => stations.len() - 1,
                "long_press" => break,
                _ => continue,
            },
            Some(Event::Media { volume: v, .. }) => {
                let pct = (v * 100.0).round() as u32;
                if volume.is_some_and(|old| old != pct) {
                    println!("  volume {pct}%");
                }
                volume = Some(pct);
                continue;
            }
            Some(_) => continue,
        };
        i = (i + step) % stations.len();
        titles.abort();
        titles = tune(&pe, &stations, i).await?;
    }

    titles.abort();
    pe.media(MediaPlayerCommand::Stop).await?;
    pe.led_off().await?;
    pe.close().await;
    println!("stopped.");
    Ok(())
}

/// Switch the device to station `i`, tint the LED ring, and start printing its titles.
async fn tune(pe: &VoicePe, stations: &[Station], i: usize) -> Result<JoinHandle<()>> {
    let s = &stations[i];
    println!("▶ {}  ({})", s.name, s.url);
    pe.play_url(&s.url, false).await?;
    pe.led(Rgb::wheel((i * 47) as u8), 0.25).await?;
    let url = s.url.clone();
    Ok(tokio::spawn(async move {
        let mut last = String::new();
        loop {
            if let Ok(Some(t)) = now_playing(&url).await
                && t != last
            {
                println!("  ♪ {t}");
                last = t;
            }
            tokio::time::sleep(Duration::from_secs(20)).await;
        }
    }))
}

/// Find MP3 stations by name on radio-browser.info, most voted first.
async fn search(name: &str) -> Result<Vec<Station>> {
    // community-run mirrors (https://api.radio-browser.info); any one may be down
    let mut last_err = None;
    for mirror in ["de1", "de2", "fi1"] {
        match search_on(mirror, name).await {
            Ok(s) => return Ok(s),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap().context("radio-browser search"))
}

async fn search_on(mirror: &str, name: &str) -> Result<Vec<Station>> {
    let found: serde_json::Value = reqwest::Client::new()
        .get(format!(
            "https://{mirror}.api.radio-browser.info/json/stations/search"
        ))
        .header("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(8))
        .query(&[
            ("name", name),
            ("codec", "MP3"),
            ("hidebroken", "true"),
            ("order", "votes"),
            ("reverse", "true"),
            ("limit", "10"),
        ])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let stations: Vec<Station> = found
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| {
            Some(Station {
                name: s["name"].as_str()?.trim().to_string(),
                url: s["url_resolved"].as_str()?.to_string(),
            })
        })
        .filter(|s| !s.url.is_empty())
        .collect();
    if stations.is_empty() {
        bail!("no MP3 stations matching {name:?}");
    }
    Ok(stations)
}

/// Read one ICY metadata block (`StreamTitle='Artist - Title';`) from the stream.
async fn now_playing(url: &str) -> Result<Option<String>> {
    let mut resp = reqwest::Client::new()
        .get(url)
        .header("Icy-MetaData", "1")
        .header("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    let Some(metaint) = resp
        .headers()
        .get("icy-metaint")
        .and_then(|v| v.to_str().ok()?.parse::<usize>().ok())
    else {
        return Ok(None); // station sends no titles
    };
    // metaint bytes of audio, one length byte (x16), then the metadata text
    let mut buf = Vec::with_capacity(metaint + 4096);
    while let Some(chunk) = resp.chunk().await? {
        buf.extend_from_slice(&chunk);
        if buf.len() > metaint {
            let end = metaint + 1 + buf[metaint] as usize * 16;
            if buf.len() >= end {
                let meta = String::from_utf8_lossy(&buf[metaint + 1..end]);
                return Ok(meta
                    .split("StreamTitle='")
                    .nth(1)
                    .and_then(|r| r.split("';").next())
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty()));
            }
        }
    }
    Ok(None)
}
