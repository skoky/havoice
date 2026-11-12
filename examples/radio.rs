//! Internet radio on the Voice PE, with volume control and spoken station names.
//!
//!     cargo run --example radio                          # preset stations
//!     cargo run --example radio -- jazz                  # search radio-browser.info (MP3 streams)
//!     cargo run --example radio -- --volume 30           # start at 30 % (0.3 works too)
//!     cargo run --example radio -- --say                 # announce each station (default voice)
//!     cargo run --example radio -- --voice Daniel jazz   # ... in a given voice
//!     cargo run --example radio -- --voice cs            # ... in the first Czech voice
//!
//! Voices come from the computer's speech program (macOS `say`, Linux `espeak-ng`);
//! `cargo run -- voices` lists them, `cargo run -- voices cs` only Czech ones.
//!
//! While it plays, type a command and press Enter:
//!     +  /  -        louder / quieter by 5 % (`+++` = 15 %)
//!     v 40           volume 40 %
//!     n  /  p        next / previous station, or a station number
//!     q              stop and quit (Ctrl-C too)
//! On the device: double press = next, triple press = previous, long press = stop;
//! the dial changes the volume as usual.
//!
//! The device fetches the stream itself (HTTP or HTTPS) - this computer only tells it which URL.
//! The firmware decodes MP3 and FLAC, so AAC / HLS / Ogg stations won't play.
//! "Now playing" comes from the stream's ICY metadata, read here.

use std::time::Duration;

use anyhow::{Result, bail};
use clap::Parser;
use havoice::{Event, MediaPlayerCommand, Rgb, VoicePe, tts};
use tokio::io::AsyncBufReadExt;
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

/// Volume step for `+` / `-`, in percent.
const STEP: i32 = 5;

struct Station {
    name: String,
    url: String,
}

/// How station names are announced: `None` = silent, `Some(None)` = default voice.
type Announce = Option<Option<String>>;

/// Internet radio on the Voice PE, with volume control and spoken station names.
/// While it plays, type + / - / v 40 (volume), n / p / a number (station), q (quit).
#[derive(Parser)]
struct Options {
    /// Start volume: 30 or 0.3 both mean 30 %
    #[arg(long, value_parser = parse_volume)]
    volume: Option<f32>,
    /// Announce each station in this voice: a name (Daniel) or a language (cs)
    #[arg(long)]
    voice: Option<String>,
    /// Announce each station in the default voice
    #[arg(long)]
    say: bool,
    /// Search radio-browser.info for these words (default: the preset stations)
    query: Vec<String>,
}

/// `30`, `30%` and `0.3` all mean 30 %.
fn parse_volume(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .trim_end_matches('%')
        .parse()
        .map_err(|_| format!("{s:?} is not a number, e.g. 30 or 0.3"))?;
    Ok(if v > 1.0 { v / 100.0 } else { v }.clamp(0.0, 1.0))
}

enum Action {
    Step(isize),
    Goto(usize),
    Louder(i32),
    SetVolume(u32),
    Help,
    Quit,
    Nothing,
}

#[tokio::main]
async fn main() -> Result<()> {
    let opts = Options::parse();

    // check the voice before anything plays, so a typo fails fast with a hint
    let announce: Announce = match &opts.voice {
        Some(v) => {
            let voice = tts::find_voice(v).await?;
            println!("voice: {} ({})", voice.name, voice.language);
            Some(Some(voice.name))
        }
        None => opts.say.then_some(None),
    };

    let stations = if opts.query.is_empty() {
        PRESETS
            .iter()
            .map(|(n, u)| Station {
                name: n.to_string(),
                url: u.to_string(),
            })
            .collect()
    } else {
        search(&opts.query.join(" ")).await?
    };
    for (i, s) in stations.iter().enumerate() {
        println!("{:2}. {}", i + 1, s.name);
    }

    let pe = VoicePe::from_env().await?;
    let mut events = pe.events().await?;
    if let Some(v) = opts.volume {
        pe.set_volume(v).await?;
    }
    println!();
    help();

    let mut i = 0;
    let mut titles = tune(&pe, &stations, i, &announce).await?;
    let mut volume: Option<u32> = None;
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin()).lines();
    let mut stdin_open = true;
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        let action = tokio::select! {
            _ = &mut ctrl_c => Action::Quit,
            line = stdin.next_line(), if stdin_open => match line {
                Ok(Some(l)) => command(&l, stations.len()),
                _ => {
                    stdin_open = false; // e.g. started in the background
                    Action::Nothing
                }
            },
            ev = events.next() => match ev {
                None => {
                    eprintln!("connection closed: {}", pe.client().close_reason());
                    Action::Quit
                }
                Some(Event::Button(press)) => match press.as_str() {
                    "double_press" => Action::Step(1),
                    "triple_press" => Action::Step(-1),
                    "long_press" => Action::Quit,
                    _ => Action::Nothing,
                },
                // the device reports every change: the dial, `+`/`-` here, Home Assistant
                Some(Event::Media { volume: v, .. }) => {
                    let pct = (v * 100.0).round() as u32;
                    if volume != Some(pct) {
                        println!("  volume {pct}%");
                    }
                    volume = Some(pct);
                    Action::Nothing
                }
                Some(_) => Action::Nothing,
            },
        };
        let next = match action {
            Action::Quit => break,
            Action::Nothing => continue,
            Action::Help => {
                help();
                continue;
            }
            Action::Louder(d) => {
                let pct = (volume.unwrap_or(50) as i32 + d).clamp(0, 100);
                pe.set_volume(pct as f32 / 100.0).await?;
                continue;
            }
            Action::SetVolume(pct) => {
                pe.set_volume(pct.min(100) as f32 / 100.0).await?;
                continue;
            }
            Action::Step(d) => (i as isize + d).rem_euclid(stations.len() as isize) as usize,
            Action::Goto(k) => k,
        };
        i = next;
        titles.abort();
        titles = tune(&pe, &stations, i, &announce).await?;
    }

    titles.abort();
    pe.media(MediaPlayerCommand::Stop).await?;
    pe.led_off().await?;
    pe.close().await;
    println!("stopped.");
    Ok(())
}

/// One line typed while the radio plays.
fn command(line: &str, stations: usize) -> Action {
    let l = line.trim().to_lowercase();
    if !l.is_empty() && l.chars().all(|c| c == '+') {
        return Action::Louder(STEP * l.len() as i32);
    }
    if !l.is_empty() && l.chars().all(|c| c == '-') {
        return Action::Louder(-STEP * l.len() as i32);
    }
    match l.as_str() {
        "" => Action::Nothing,
        "n" | "next" => Action::Step(1),
        "p" | "prev" => Action::Step(-1),
        "q" | "quit" | "stop" => Action::Quit,
        _ => {
            if let Some(Ok(pct)) = l
                .strip_prefix('v')
                .map(|v| v.trim().trim_end_matches('%').parse::<u32>())
            {
                return Action::SetVolume(pct);
            }
            match l.parse::<usize>() {
                Ok(k) if (1..=stations).contains(&k) => Action::Goto(k - 1),
                _ => Action::Help,
            }
        }
    }
}

fn help() {
    println!(
        "type + / - (volume ±{STEP} %), v 40 (volume 40 %), n / p or a number (station), q (quit)\n\
         device: double press = next, triple = previous, long press = stop, dial = volume\n"
    );
}

/// Switch the device to station `i`, tint the LED ring, and start printing its titles.
/// With `announce`, the station name is spoken first (over the old station, which ducks).
async fn tune(
    pe: &VoicePe,
    stations: &[Station],
    i: usize,
    announce: &Announce,
) -> Result<JoinHandle<()>> {
    let s = &stations[i];
    println!("▶ {}. {}  ({})", i + 1, s.name, s.url);
    pe.led(Rgb::wheel((i * 47) as u8), 0.25).await?;
    if let Some(voice) = announce {
        let text = spoken_name(&s.name);
        let said = match voice {
            Some(v) => pe.say_voice(&text, v).await,
            None => pe.say(&text).await,
        };
        if let Err(e) = said {
            eprintln!("  (couldn't announce: {e:#})");
        }
    }
    pe.play_url(&s.url, false).await?;
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

/// Station names from the directory carry clutter: "SomaFM Groove Salad (128k MP3)",
/// "Dance Wave! | MP3". Speak only the part before it.
fn spoken_name(name: &str) -> String {
    let short = name
        .split(['(', '|', '[', ','])
        .next()
        .unwrap_or(name)
        .trim();
    let short = if short.is_empty() { name } else { short };
    short.chars().take(60).collect()
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
