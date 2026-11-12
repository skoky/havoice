//! High-level handle for a Home Assistant Voice PE.
//!
//! ```no_run
//! # async fn demo() -> anyhow::Result<()> {
//! use havoice::{Event, Rgb, VoicePe};
//! let pe = VoicePe::from_env().await?;          // HAVOICE_HOST / HAVOICE_KEY or .env
//! pe.led(Rgb(255, 100, 0), 0.8).await?;          // LED ring orange
//! pe.say("Hello from the Mac").await?;           // say / espeak-ng -> speaker
//! let mut events = pe.events().await?;
//! while let Some(ev) = events.next().await {
//!     if let Event::Button(press) = ev { println!("button: {press}"); }
//! }
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use esphome_client::types as pb;
use esphome_client::types::EspHomeMessage as Msg;
use tokio::sync::broadcast;

use crate::api::{Client, Entity};
use crate::audio::Pcm;
use crate::{discover, serve};

pub use pb::MediaPlayerCommand;

/// 8-bit RGB colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    /// Parse `RRGGBB` or `#RRGGBB`.
    pub fn hex(s: &str) -> Result<Self> {
        let v =
            u32::from_str_radix(s.trim_start_matches('#'), 16).context("colour must be RRGGBB")?;
        Ok(Self((v >> 16) as u8, (v >> 8) as u8, v as u8))
    }

    /// Colour wheel, `pos` 0..=255.
    pub fn wheel(pos: u8) -> Self {
        match pos {
            0..=84 => Self(255 - pos * 3, pos * 3, 0),
            85..=169 => Self(0, 255 - (pos - 85) * 3, (pos - 85) * 3),
            _ => Self((pos - 170) * 3, 0, 255 - (pos - 170) * 3),
        }
    }
}

/// Something that happened on the device, already named.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Centre button: `double_press`, `triple_press`, `long_press`, `easter_egg_press`
    /// (a single press is not reported: the firmware uses it for the voice assistant).
    Button(String),
    /// Media player changed (the dial changes `volume`).
    Media {
        state: String,
        volume: f32,
        muted: bool,
    },
    Switch {
        name: String,
        on: bool,
    },
    Select {
        name: String,
        value: String,
    },
    Light {
        name: String,
        on: bool,
        rgb: Rgb,
        brightness: f32,
        effect: String,
    },
    BinarySensor {
        name: String,
        on: bool,
    },
    Sensor {
        name: String,
        value: f32,
    },
    Text {
        name: String,
        value: String,
    },
    Number {
        name: String,
        value: f32,
    },
    /// Any other `event` entity.
    Other {
        name: String,
        kind: String,
    },
}

pub struct VoicePe {
    client: Client,
    ip: IpAddr,
    entities: Vec<Entity>,
}

impl VoicePe {
    /// Connect to `host` (name or IP, optional `:port`) with an optional Noise key.
    /// A device that doesn't answer (Wi-Fi asleep, still booting) is retried a few times.
    pub async fn connect(host: &str, key: Option<&str>) -> Result<Self> {
        let addr = resolve(host).await?;
        let client = connect_retrying(addr, key).await?;
        let entities = client.list_entities().await?;
        Ok(Self {
            client,
            ip: addr.ip(),
            entities,
        })
    }

    /// First-time setup without Home Assistant. On a device that has no API key yet (new,
    /// or factory reset by holding the centre button ~22 s), generate a key, store it on
    /// the device and reconnect encrypted. Returns the connection and the new base64 key -
    /// save it, it's the only copy.
    pub async fn pair(host: &str) -> Result<(Self, String)> {
        let addr = resolve(host).await?;
        let plain = connect_retrying(addr, None).await.map_err(|e| {
            anyhow!("{}", e.root_cause()).context(
                "device refused a plaintext connection, so it already has a key: factory \
                     reset it (hold the centre button ~22 s until the red circle completes), \
                     or rotate the key with the current one",
            )
        })?;
        let info = plain.device_info().await?;
        anyhow::ensure!(
            info.api_encryption_supported,
            "this firmware can't take an API key over the air"
        );
        let key = crate::api::generate_key()?;
        plain.set_noise_key(&key).await?;
        let pe = Self::reconnect(host, &key).await?;
        Ok((pe, key))
    }

    /// Replace the key on a paired device with a fresh one; returns the new connection and
    /// key. Everything else using the old key - Home Assistant included - is disconnected
    /// and needs the new key.
    pub async fn rotate_key(self) -> Result<(Self, String)> {
        let host = self.ip.to_string();
        let key = crate::api::generate_key()?;
        self.client.set_noise_key(&key).await?;
        let pe = Self::reconnect(&host, &key).await?;
        Ok((pe, key))
    }

    /// The device applies a new key ~100 ms after confirming it, then drops all clients.
    async fn reconnect(host: &str, key: &str) -> Result<Self> {
        let mut last = None;
        for _ in 0..10 {
            tokio::time::sleep(Duration::from_millis(500)).await;
            match Self::connect(host, Some(key)).await {
                Ok(pe) => return Ok(pe),
                Err(e) => last = Some(e),
            }
        }
        Err(last
            .unwrap()
            .context("device took the key but won't accept it - retry with it"))
    }

    /// Connect using `HAVOICE_HOST` (else mDNS) and `HAVOICE_KEY`, reading `.env` if present.
    pub async fn from_env() -> Result<Self> {
        load_env();
        let host = match std::env::var("HAVOICE_HOST").ok().filter(|h| !h.is_empty()) {
            Some(h) => h,
            None => find_on_lan().await?,
        };
        let key = std::env::var("HAVOICE_KEY").ok().filter(|k| !k.is_empty());
        Self::connect(&host, key.as_deref()).await
    }

    /// The underlying API client, for anything not wrapped here.
    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// First entity of `kind` whose object_id contains `needle`.
    pub fn entity(&self, kind: &str, needle: &str) -> Result<&Entity> {
        self.entities
            .iter()
            .find(|e| e.kind == kind && e.object_id.contains(needle))
            .ok_or_else(|| {
                let have: Vec<_> = self
                    .entities
                    .iter()
                    .filter(|e| e.kind == kind)
                    .map(|e| e.object_id.as_str())
                    .collect();
                anyhow!("no {kind} matching {needle:?}; available: {have:?}")
            })
    }

    pub async fn info(&self) -> Result<pb::DeviceInfoResponse> {
        self.client.device_info().await
    }

    // ---------------------------------------------------------------- LED ring

    fn ring(&self) -> Result<&Entity> {
        self.entity("light", "led_ring")
    }

    /// Light effects the LED ring offers.
    pub fn led_effects(&self) -> Vec<String> {
        self.ring().map(|e| e.choices.clone()).unwrap_or_default()
    }

    /// Whole ring one colour, `brightness` 0.0..=1.0.
    pub async fn led(&self, rgb: Rgb, brightness: f32) -> Result<()> {
        self.led_cmd(|r| {
            r.has_state = true;
            r.state = true;
            r.has_rgb = true;
            r.red = rgb.0 as f32 / 255.0;
            r.green = rgb.1 as f32 / 255.0;
            r.blue = rgb.2 as f32 / 255.0;
            r.has_brightness = true;
            r.brightness = brightness;
            r.has_effect = true;
            r.effect = "None".into();
        })
        .await
    }

    /// Run one of [`Self::led_effects`].
    pub async fn led_effect(&self, effect: &str) -> Result<()> {
        self.led_cmd(|r| {
            r.has_state = true;
            r.state = true;
            r.has_effect = true;
            r.effect = effect.into();
        })
        .await
    }

    /// Fade to a colour over `secs`.
    pub async fn led_fade(&self, rgb: Rgb, brightness: f32, secs: f32) -> Result<()> {
        self.led_cmd(|r| {
            r.has_state = true;
            r.state = true;
            r.has_rgb = true;
            r.red = rgb.0 as f32 / 255.0;
            r.green = rgb.1 as f32 / 255.0;
            r.blue = rgb.2 as f32 / 255.0;
            r.has_brightness = true;
            r.brightness = brightness;
            r.has_transition_length = true;
            r.transition_length = (secs * 1000.0) as u32;
        })
        .await
    }

    pub async fn led_off(&self) -> Result<()> {
        self.led_cmd(|r| {
            r.has_state = true;
            r.state = false;
        })
        .await
    }

    async fn led_cmd(&self, f: impl FnOnce(&mut pb::LightCommandRequest)) -> Result<()> {
        let mut r = pb::LightCommandRequest {
            key: self.ring()?.key,
            ..Default::default()
        };
        f(&mut r);
        self.client.send(r).await
    }

    // ---------------------------------------------------------------- audio

    fn player(&self) -> Result<&Entity> {
        self.entity("media_player", "")
    }

    async fn media_cmd(&self, f: impl FnOnce(&mut pb::MediaPlayerCommandRequest)) -> Result<()> {
        let mut r = pb::MediaPlayerCommandRequest {
            key: self.player()?.key,
            ..Default::default()
        };
        f(&mut r);
        self.client.send(r).await
    }

    /// Stream an http(s) URL. The stock firmware decodes FLAC and MP3 (not WAV). `announce` ducks music and plays on top.
    pub async fn play_url(&self, url: &str, announce: bool) -> Result<()> {
        self.media_cmd(|r| {
            r.has_media_url = true;
            r.media_url = url.into();
            r.has_announcement = true;
            r.announcement = announce;
        })
        .await
    }

    /// Serve a local file from this Mac and play it. The file is served for as long as
    /// this process runs; returns the URL handed to the device.
    pub async fn play_file(&self, path: impl AsRef<Path>, announce: bool) -> Result<String> {
        let path = std::fs::canonicalize(path.as_ref())
            .with_context(|| format!("no such file {}", path.as_ref().display()))?;
        let url = serve::serve_file(path, self.ip).await?;
        self.play_url(&url, announce).await?;
        Ok(url)
    }

    /// Play generated audio: encodes to FLAC, serves it, and returns once the device
    /// should have finished playing it.
    pub async fn play_pcm(&self, pcm: &Pcm, announce: bool) -> Result<()> {
        let url = serve::serve_bytes(pcm.to_flac()?, "havoice.flac", self.ip).await?;
        self.play_url(&url, announce).await?;
        // fetch + decoder start-up + playback
        tokio::time::sleep(Duration::from_secs_f32(pcm.seconds() + 1.5)).await;
        Ok(())
    }

    /// Speak `text` on the device as an announcement (music ducks underneath) and wait until
    /// it's done. Uses macOS `say` / `espeak-ng` and the default voice - see [`crate::tts`].
    pub async fn say(&self, text: &str) -> Result<()> {
        self.play_pcm(&crate::tts::render(text, None).await?, true)
            .await
    }

    /// [`say`](Self::say) in a given voice: a name such as `Zuzana`, or as passed to the
    /// speech program's `-v`. Use [`crate::tts::find_voice`] to pick one by language.
    pub async fn say_voice(&self, text: &str, voice: &str) -> Result<()> {
        self.play_pcm(&crate::tts::render(text, Some(voice)).await?, true)
            .await
    }

    pub async fn media(&self, cmd: MediaPlayerCommand) -> Result<()> {
        self.media_cmd(|r| {
            r.has_command = true;
            r.command = cmd as i32;
        })
        .await
    }

    /// Volume 0.0..=1.0.
    pub async fn set_volume(&self, volume: f32) -> Result<()> {
        self.media_cmd(|r| {
            r.has_volume = true;
            r.volume = volume.clamp(0.0, 1.0);
        })
        .await
    }

    // ---------------------------------------------------------------- settings

    pub async fn select(&self, needle: &str, option: &str) -> Result<()> {
        let e = self.entity("select", needle)?;
        if !e.choices.is_empty() && !e.choices.iter().any(|c| c == option) {
            return Err(anyhow!("{option:?} is not one of {:?}", e.choices));
        }
        self.client
            .send(pb::SelectCommandRequest {
                key: e.key,
                state: option.into(),
                ..Default::default()
            })
            .await
    }

    pub async fn switch(&self, needle: &str, on: bool) -> Result<()> {
        let e = self.entity("switch", needle)?;
        self.client
            .send(pb::SwitchCommandRequest {
                key: e.key,
                state: on,
                ..Default::default()
            })
            .await
    }

    pub async fn number(&self, needle: &str, value: f32) -> Result<()> {
        let e = self.entity("number", needle)?;
        self.client
            .send(pb::NumberCommandRequest {
                key: e.key,
                state: value,
                ..Default::default()
            })
            .await
    }

    pub async fn press(&self, needle: &str) -> Result<()> {
        let e = self.entity("button", needle)?;
        self.client
            .send(pb::ButtonCommandRequest {
                key: e.key,
                ..Default::default()
            })
            .await
    }

    /// Wake words the firmware offers (e.g. "Okay Nabu", "Hey Jarvis", "Hey Mycroft").
    pub fn wake_words(&self) -> Vec<String> {
        self.entity("select", "wake_word")
            .map(|e| e.choices.clone())
            .unwrap_or_default()
    }

    pub async fn set_wake_word(&self, word: &str) -> Result<()> {
        self.select("wake_word", word).await
    }

    /// Turn the "beep" after the wake word on/off.
    pub async fn set_wake_sound(&self, on: bool) -> Result<()> {
        self.switch("wake_sound", on).await
    }

    // ---------------------------------------------------------------- events

    /// Stream of named device events. The first batch is the current state of everything.
    pub async fn events(&self) -> Result<Events> {
        let rx = self.client.subscribe();
        self.client.subscribe_states().await?;
        let names = self.entities.iter().map(|e| (e.key, e.clone())).collect();
        Ok(Events { rx, names })
    }

    pub async fn close(self) {
        self.client.disconnect().await;
    }
}

pub struct Events {
    rx: broadcast::Receiver<Msg>,
    names: HashMap<u32, Entity>,
}

impl Events {
    /// Next event; `None` once the connection is gone.
    pub async fn next(&mut self) -> Option<Event> {
        loop {
            let msg = match self.rx.recv().await {
                Ok(m) => m,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => return None,
            };
            if let Some(ev) = self.convert(&msg) {
                return Some(ev);
            }
        }
    }

    fn name(&self, key: u32) -> String {
        self.names
            .get(&key)
            .map_or_else(|| key.to_string(), |e| e.object_id.clone())
    }

    fn convert(&self, msg: &Msg) -> Option<Event> {
        Some(match msg {
            Msg::EventResponse(m) => {
                let name = self.name(m.key);
                if name.contains("button") {
                    Event::Button(m.event_type.clone())
                } else {
                    Event::Other {
                        name,
                        kind: m.event_type.clone(),
                    }
                }
            }
            Msg::MediaPlayerStateResponse(m) => Event::Media {
                state: pb::MediaPlayerState::try_from(m.state)
                    .map_or_else(|_| m.state.to_string(), |s| s.as_str_name().to_lowercase()),
                volume: m.volume,
                muted: m.muted,
            },
            Msg::SwitchStateResponse(m) => Event::Switch {
                name: self.name(m.key),
                on: m.state,
            },
            Msg::SelectStateResponse(m) => Event::Select {
                name: self.name(m.key),
                value: m.state.clone(),
            },
            Msg::LightStateResponse(m) => Event::Light {
                name: self.name(m.key),
                on: m.state,
                rgb: Rgb(
                    (m.red * 255.0) as u8,
                    (m.green * 255.0) as u8,
                    (m.blue * 255.0) as u8,
                ),
                brightness: m.brightness,
                effect: m.effect.clone(),
            },
            Msg::BinarySensorStateResponse(m) => Event::BinarySensor {
                name: self.name(m.key),
                on: m.state,
            },
            Msg::SensorStateResponse(m) => Event::Sensor {
                name: self.name(m.key),
                value: m.state,
            },
            Msg::TextSensorStateResponse(m) => Event::Text {
                name: self.name(m.key),
                value: m.state.clone(),
            },
            Msg::NumberStateResponse(m) => Event::Number {
                name: self.name(m.key),
                value: m.state,
            },
            _ => return None,
        })
    }
}

// -------------------------------------------------------------------- helpers

/// Load `.env` from the crate dir and the current dir (existing env vars win).
pub fn load_env() {
    let _ = dotenvy::from_path(concat!(env!("CARGO_MANIFEST_DIR"), "/.env"));
    let _ = dotenvy::dotenv();
}

/// Hostname of the first Voice PE announced on the LAN.
pub async fn find_on_lan() -> Result<String> {
    discover::discover(Duration::from_secs(3))
        .await?
        .into_iter()
        .find(|d| d.name.contains("voice"))
        .map(|d| d.host)
        .ok_or_else(|| anyhow!("no Voice PE found via mDNS; set HAVOICE_HOST"))
}

/// `host` (`name`, `name:port`, `ip` or `ip:port`) -> IPv4 address of the API port.
/// Names go to the system resolver first; for `.local` names that come back without an
/// IPv4 address (macOS does that now and then), the device is asked over mDNS instead.
async fn resolve(host: &str) -> Result<SocketAddr> {
    let (name, port) = match host.rsplit_once(':') {
        Some((n, p)) if !n.contains(':') => (
            n,
            p.parse::<u16>()
                .with_context(|| format!("bad port in {host:?}"))?,
        ),
        _ => (host, 6053),
    };
    if let Ok(ip) = name.parse::<IpAddr>() {
        return Ok(SocketAddr::new(ip, port));
    }
    let system = tokio::net::lookup_host((name, port))
        .await
        .map(|addrs| addrs.collect::<Vec<_>>());
    if let Some(&a) = system.iter().flatten().find(|a| a.is_ipv4()) {
        return Ok(a);
    }
    if name.trim_end_matches('.').ends_with(".local")
        && let Some(ip) = discover::resolve_v4(name, Duration::from_secs(3)).await?
    {
        return Ok(SocketAddr::new(ip.into(), port));
    }
    Err(match system {
        Err(e) => anyhow!(e).context(format!("resolving {name}")),
        Ok(_) => anyhow!("no IPv4 address for {name} (is the device on?)"),
    })
}

const CONNECT_TRIES: u32 = 3;

/// `Client::connect`, retried when the device simply didn't answer. Wrong keys and
/// protocol errors fail straight away.
async fn connect_retrying(addr: SocketAddr, key: Option<&str>) -> Result<Client> {
    let mut tries = 0;
    loop {
        tries += 1;
        match Client::connect(&addr.to_string(), key, "havoice").await {
            Ok(c) => return Ok(c),
            Err(e) if tries < CONNECT_TRIES && transient(&e) => {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            Err(e) if unreachable(&e) => {
                return Err(e.context(format!(
                    "{} is unreachable - if `ping {}` works, macOS is blocking this app: \
                     allow your terminal in System Settings > Privacy & Security > Local Network",
                    addr.ip(),
                    addr.ip()
                )));
            }
            Err(e) => return Err(e),
        }
    }
}

fn io_kind(e: &anyhow::Error) -> Option<std::io::ErrorKind> {
    use esphome_client::error::{ClientError, ConnectionError};
    match e.downcast_ref::<ClientError>()? {
        ClientError::Connection(ConnectionError::TcpConnect { source, .. }) => Some(source.kind()),
        _ => None,
    }
}

/// No answer, or the connection was refused/reset while the device was busy or waking.
fn transient(e: &anyhow::Error) -> bool {
    use esphome_client::error::ClientError;
    use std::io::ErrorKind::*;
    matches!(
        e.downcast_ref::<ClientError>(),
        Some(ClientError::Timeout { .. })
    ) || matches!(
        io_kind(e),
        Some(TimedOut | ConnectionRefused | ConnectionReset | HostUnreachable | NetworkUnreachable)
    )
}

fn unreachable(e: &anyhow::Error) -> bool {
    use std::io::ErrorKind::*;
    matches!(io_kind(e), Some(HostUnreachable | NetworkUnreachable))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resolve_takes_ips_names_and_ports() {
        let a = resolve("10.0.10.79").await.unwrap();
        assert_eq!(a.to_string(), "10.0.10.79:6053");
        let a = resolve("127.0.0.1:7000").await.unwrap();
        assert_eq!(a.to_string(), "127.0.0.1:7000");
        let a = resolve("localhost:7000").await.unwrap();
        assert_eq!(a.to_string(), "127.0.0.1:7000");
        assert!(resolve("10.0.10.79:http").await.is_err());
    }

    #[tokio::test]
    async fn refused_connections_are_retried_then_reported() {
        // bind and drop: nothing listens on this port any more
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let start = std::time::Instant::now();
        let err = VoicePe::connect(&format!("127.0.0.1:{port}"), None)
            .await
            .err()
            .expect("nothing listens there");
        assert!(transient(&err), "{err:#}");
        // CONNECT_TRIES attempts, 1 s apart
        assert!(start.elapsed() >= Duration::from_secs(u64::from(CONNECT_TRIES - 1)));
    }
}
