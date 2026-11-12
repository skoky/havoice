//! Set up a Voice PE without Home Assistant: Wi-Fi over the USB cable, then a fresh API key.
//!
//!     cargo run --example pair -- --status          # read-only: Wi-Fi + key state
//!     cargo run --example pair -- --wifi "MySSID"   # Wi-Fi over USB (asks for the password), then key
//!     cargo run --example pair                      # already on Wi-Fi: just set the key
//!     cargo run --example pair -- --rotate          # paired already: replace HAVOICE_KEY
//!
//! Pairing needs a device without a key: a new one, or one you factory reset by holding the
//! centre button ~22 s until the red circle completes (this erases Wi-Fi, the API key and LED
//! settings; a white twinkle means it's ready). A device still adopted by Home Assistant
//! loses that connection - remove it from HA first, or re-add it there with the new key.
//!
//! The new key goes into `.env` as HAVOICE_KEY (the previous file is kept as `.env.bak`).
//! The Wi-Fi password is read from HAVOICE_WIFI_PASSWORD or asked for; it's never stored.

use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use havoice::device::{find_on_lan, load_env};
use havoice::improv::{Improv, State};
use havoice::{Rgb, VoicePe, discover, serial};

/// Set up a Voice PE without Home Assistant: Wi-Fi over USB, then a fresh API key.
#[derive(Parser)]
struct Args {
    /// Read-only: Wi-Fi state over USB and whether the device has a key
    #[arg(long, conflicts_with_all = ["wifi", "rotate"])]
    status: bool,
    /// Send Wi-Fi settings over USB first (password: HAVOICE_WIFI_PASSWORD or prompt)
    #[arg(long, value_name = "SSID")]
    wifi: Option<String>,
    /// Replace the key of an already paired device (cuts off Home Assistant)
    #[arg(long)]
    rotate: bool,
    /// USB serial port (default: the Espressif one)
    #[arg(long, env = "HAVOICE_SERIAL")]
    port: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    load_env();
    let Args {
        status,
        wifi,
        rotate,
        port,
    } = Args::parse();
    let port = || -> Result<String> {
        match &port {
            Some(p) => Ok(p.clone()),
            None => serial::default_port(),
        }
    };

    if status {
        return show_status(port().ok()).await;
    }

    // 1. Wi-Fi over USB (Improv Serial)
    let mut host = None;
    if let Some(ssid) = wifi {
        let password = match std::env::var("HAVOICE_WIFI_PASSWORD") {
            Ok(p) => p,
            Err(_) => rpassword::prompt_password(format!("Wi-Fi password for {ssid:?}: "))?,
        };
        let port = port()?;
        println!("sending Wi-Fi settings over {port} ...");
        let url = tokio::task::spawn_blocking(move || {
            let mut improv = Improv::open(&port)?;
            improv.provision(&ssid, &password, Duration::from_secs(60))
        })
        .await??;
        println!(
            "on Wi-Fi{}",
            url.as_deref().map(|u| format!(": {u}")).unwrap_or_default()
        );
        host = url.and_then(|u| host_of(&u));
    }

    // 2. API key
    let (pe, key) = if rotate {
        println!("replacing the key - anything using the old one (Home Assistant too) is cut off");
        VoicePe::from_env().await?.rotate_key().await?
    } else {
        let host = match host.or_else(|| std::env::var("HAVOICE_HOST").ok()) {
            Some(h) if !h.is_empty() => h,
            _ => find_on_lan().await?,
        };
        println!("pairing with {host} ...");
        let (pe, key) = VoicePe::pair(&host).await?;
        save_env(&host, &key)?;
        (pe, key)
    };
    if rotate {
        save_env(&std::env::var("HAVOICE_HOST").unwrap_or_default(), &key)?;
    }
    println!("new key stored on the device and saved to .env as HAVOICE_KEY");

    // 3. Say hello over the new encrypted connection
    let info = pe.info().await?;
    println!(
        "connected (encrypted) to {} - {}",
        info.friendly_name, info.project_version
    );
    pe.led(Rgb(0, 255, 60), 0.6).await?;
    pe.say("Paired. Hello from Rust!").await?;
    pe.led_off().await?;
    pe.close().await;
    Ok(())
}

/// Read-only look at the device: Improv over USB, key state via mDNS.
async fn show_status(port: Option<String>) -> Result<()> {
    if let Some(port) = port {
        let (info, state) = tokio::task::spawn_blocking(move || -> Result<_> {
            let mut improv = Improv::open(&port)?;
            Ok((improv.info()?, improv.state()?))
        })
        .await??;
        println!("USB      {}", info.join(" / "));
        match state {
            (State::Provisioned, url) => {
                println!("Wi-Fi    connected {}", url.unwrap_or_default())
            }
            (s, _) => println!("Wi-Fi    not set up ({s:?}) - use --wifi SSID"),
        }
    } else {
        println!("USB      no device on USB - skipping Wi-Fi check");
    }
    let found = discover::discover(Duration::from_secs(3)).await?;
    match found.iter().find(|d| d.name.contains("voice")) {
        Some(d) if d.encrypted => println!(
            "API key  set on {} - pair needs a factory reset, or use --rotate",
            d.host
        ),
        Some(d) => println!("API key  none on {} - ready to pair", d.host),
        None => println!("API key  device not found on the LAN"),
    }
    Ok(())
}

/// `http://10.0.10.76` / `http://10.0.10.76:80/` -> `10.0.10.76`
fn host_of(url: &str) -> Option<String> {
    let rest = url.split("://").nth(1)?;
    let host = rest.split(['/', ':']).next()?;
    (!host.is_empty()).then(|| host.to_string())
}

/// Set HAVOICE_HOST / HAVOICE_KEY in the crate's `.env`, keeping everything else.
fn save_env(host: &str, key: &str) -> Result<()> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/.env");
    let old = std::fs::read_to_string(path).unwrap_or_default();
    if !old.is_empty() {
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/.env.bak"), &old)?;
    }
    let mut lines: Vec<String> = old
        .lines()
        .filter(|l| {
            let l = l.trim_start().trim_start_matches("export ");
            !l.starts_with("HAVOICE_KEY=") && !(l.starts_with("HAVOICE_HOST=") && !host.is_empty())
        })
        .map(String::from)
        .collect();
    if !host.is_empty() {
        lines.push(format!("HAVOICE_HOST={host}"));
    }
    lines.push(format!("HAVOICE_KEY={key}"));
    std::fs::write(path, lines.join("\n") + "\n")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}
