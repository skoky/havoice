//! `havoice` - poke at a Home Assistant Voice Preview Edition from the Mac.
//! Thin CLI over the `havoice` library (see examples/ for programmatic use).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand, ValueEnum};
use havoice::device::{find_on_lan, load_env};
use havoice::proto::{self as pb, EspHomeMessage};
use havoice::{Event, MediaPlayerCommand as M, Rgb, VoicePe, discover, serial};

#[derive(Parser)]
#[command(name = "havoice", about = "Play with the Home Assistant Voice PE")]
struct Cli {
    /// Device hostname or IP (default: first Voice PE found via mDNS)
    #[arg(long, env = "HAVOICE_HOST", global = true)]
    host: Option<String>,
    /// Base64 Noise PSK (api encryption key) - see README
    #[arg(long, env = "HAVOICE_KEY", global = true, hide_env_values = true)]
    key: Option<String>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Find ESPHome devices on the LAN via mDNS
    Discover {
        #[arg(long, default_value_t = 3)]
        secs: u64,
    },
    /// Tail the USB serial log
    Serial {
        /// Serial port (default: first /dev/cu.usbmodem*)
        #[arg(long, env = "HAVOICE_SERIAL")]
        port: Option<String>,
        /// Only lines matching this regex (case-insensitive)
        #[arg(long)]
        grep: Option<String>,
        /// Keep ANSI colours
        #[arg(long)]
        raw: bool,
        /// Also append plain lines to this file
        #[arg(long)]
        save: Option<PathBuf>,
    },
    /// Device info over the API
    Info,
    /// List all entities
    Entities,
    /// Stream events: button, dial, mute switch, media player, LEDs ...
    Watch,
    /// Device logs over Wi-Fi
    Logs {
        #[arg(long, value_enum, default_value_t = Level::Debug)]
        level: Level,
    },
    /// LED ring: RRGGBB hex, an effect name, or "off"
    Led {
        color: String,
        #[arg(long, default_value_t = 0.8)]
        brightness: f32,
    },
    /// Speak text with the Mac's `say` voice
    Say { text: Vec<String> },
    /// Play an http(s) URL or a local mp3/flac/wav file
    Play {
        src: String,
        /// Play as an announcement (ducks music)
        #[arg(long)]
        announce: bool,
        #[arg(long)]
        volume: Option<f32>,
        /// Seconds to keep serving a local file
        #[arg(long, default_value_t = 30)]
        hold: u64,
    },
    /// Media player control
    Media {
        #[arg(value_enum)]
        action: MediaAction,
        /// Volume 0.0-1.0 for `volume`
        value: Option<f32>,
    },
    /// Set a select, e.g. `select wake_word "Hey Jarvis"`
    Select { entity: String, option: String },
    /// Set a switch, e.g. `switch wake_sound off`
    Switch { entity: String, state: OnOff },
    /// Set a number entity
    Number { entity: String, value: f32 },
    /// Press a button entity
    Press { entity: String },
}

#[derive(Clone, Copy, ValueEnum)]
enum Level {
    Error,
    Warn,
    Info,
    Config,
    Debug,
    Verbose,
}

#[derive(Clone, Copy, ValueEnum)]
enum MediaAction {
    Play,
    Pause,
    Stop,
    Mute,
    Unmute,
    Toggle,
    VolumeUp,
    VolumeDown,
    Volume,
}

#[derive(Clone, Copy, ValueEnum)]
enum OnOff {
    On,
    Off,
}

#[tokio::main]
async fn main() -> Result<()> {
    load_env();
    let cli = Cli::parse();

    // commands that don't need the API
    match cli.cmd {
        Cmd::Discover { secs } => {
            for d in discover::discover(Duration::from_secs(secs)).await? {
                let enc = if d.encrypted {
                    "encrypted"
                } else {
                    "plaintext"
                };
                println!("{}  {}:{}  ({enc})", d.name, d.host, d.port);
                for k in [
                    "friendly_name",
                    "mac",
                    "version",
                    "project_name",
                    "project_version",
                ] {
                    if let Some(v) = d.txt.get(k) {
                        println!("    {k:16} {v}");
                    }
                }
            }
            return Ok(());
        }
        Cmd::Serial {
            port,
            grep,
            raw,
            save,
        } => {
            let port = match port {
                Some(p) => p,
                None => serial::default_port()?,
            };
            let grep = grep
                .map(|g| regex::Regex::new(&format!("(?i){g}")))
                .transpose()?;
            let save = save
                .map(|p| {
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(p)
                })
                .transpose()?;
            return tokio::task::spawn_blocking(move || {
                serial::tail(&port, grep.as_ref(), raw, save)
            })
            .await?;
        }
        _ => {}
    }

    let host = match cli.host.filter(|h| !h.is_empty()) {
        Some(h) => h,
        None => find_on_lan().await?,
    };
    let key = cli.key.filter(|k| !k.is_empty());
    if key.is_none() {
        eprintln!("# HAVOICE_KEY not set - trying plaintext (the stock firmware requires the key)");
    }
    let pe = VoicePe::connect(&host, key.as_deref()).await?;

    match cli.cmd {
        Cmd::Discover { .. } | Cmd::Serial { .. } => unreachable!(),
        Cmd::Info => {
            let d = pe.info().await?;
            let h = &pe.client().hello;
            println!(
                "api               {}.{} ({})",
                h.api_version_major, h.api_version_minor, h.server_info
            );
            println!("name              {} / {}", d.name, d.friendly_name);
            println!("mac               {}", d.mac_address);
            println!(
                "esphome           {} built {}",
                d.esphome_version, d.compilation_time
            );
            println!("model             {} {}", d.manufacturer, d.model);
            println!("project           {} {}", d.project_name, d.project_version);
            println!("voice flags       {:#b}", d.voice_assistant_feature_flags);
        }
        Cmd::Entities => {
            let mut ents = pe.entities().to_vec();
            ents.sort_by(|a, b| (a.kind, &a.object_id).cmp(&(b.kind, &b.object_id)));
            for e in ents {
                let extra = if e.choices.is_empty() {
                    String::new()
                } else {
                    format!("  {:?}", e.choices)
                };
                println!("{:14} {:42} {:?}{extra}", e.kind, e.object_id, e.name);
            }
        }
        Cmd::Watch => {
            let mut events = pe.events().await?;
            eprintln!("# watching - press the button, turn the dial, flip mute. Ctrl-C to stop");
            while let Some(ev) = events.next().await {
                let t = chrono::Local::now().format("%H:%M:%S%.3f");
                match ev {
                    Event::Button(p) => println!("{t}  BUTTON {p}"),
                    other => println!("{t}  {other:?}"),
                }
            }
            return Err(anyhow!("connection closed: {}", pe.client().close_reason()));
        }
        Cmd::Logs { level } => {
            let c = pe.client();
            let mut rx = c.subscribe();
            c.subscribe_logs(match level {
                Level::Error => pb::LogLevel::Error,
                Level::Warn => pb::LogLevel::Warn,
                Level::Info => pb::LogLevel::Info,
                Level::Config => pb::LogLevel::Config,
                Level::Debug => pb::LogLevel::Debug,
                Level::Verbose => pb::LogLevel::Verbose,
            })
            .await?;
            let ansi = regex::Regex::new(r"\x1b\[[0-9;]*m")?;
            loop {
                match rx.recv().await {
                    Ok(EspHomeMessage::SubscribeLogsResponse(l)) => {
                        println!(
                            "{}",
                            ansi.replace_all(&String::from_utf8_lossy(&l.message), "")
                        )
                    }
                    Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                    Err(_) => return Err(anyhow!("connection closed: {}", c.close_reason())),
                }
            }
        }
        Cmd::Led { color, brightness } => match color.as_str() {
            "off" => pe.led_off().await?,
            c if c.trim_start_matches('#').len() == 6 && Rgb::hex(c).is_ok() => {
                pe.led(Rgb::hex(c)?, brightness).await?
            }
            effect => pe.led_effect(effect).await?,
        },
        Cmd::Say { text } => pe.say(&text.join(" ")).await?,
        Cmd::Play {
            src,
            announce,
            volume,
            hold,
        } => {
            if let Some(v) = volume {
                pe.set_volume(v).await?;
            }
            if src.starts_with("http://") || src.starts_with("https://") {
                pe.play_url(&src, announce).await?;
                println!("playing {src}");
            } else {
                println!("playing {}", pe.play_file(&src, announce).await?);
                tokio::time::sleep(Duration::from_secs(hold)).await;
            }
        }
        Cmd::Media { action, value } => match action {
            MediaAction::Volume => {
                pe.set_volume(value.ok_or_else(|| anyhow!("volume needs 0.0-1.0"))?)
                    .await?
            }
            MediaAction::Play => pe.media(M::Play).await?,
            MediaAction::Pause => pe.media(M::Pause).await?,
            MediaAction::Stop => pe.media(M::Stop).await?,
            MediaAction::Mute => pe.media(M::Mute).await?,
            MediaAction::Unmute => pe.media(M::Unmute).await?,
            MediaAction::Toggle => pe.media(M::Toggle).await?,
            MediaAction::VolumeUp => pe.media(M::VolumeUp).await?,
            MediaAction::VolumeDown => pe.media(M::VolumeDown).await?,
        },
        Cmd::Select { entity, option } => pe.select(&entity, &option).await?,
        Cmd::Switch { entity, state } => pe.switch(&entity, matches!(state, OnOff::On)).await?,
        Cmd::Number { entity, value } => pe.number(&entity, value).await?,
        Cmd::Press { entity } => pe.press(&entity).await?,
    }
    pe.close().await;
    Ok(())
}
