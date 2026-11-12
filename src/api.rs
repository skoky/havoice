//! Thin async layer over `esphome-client`: one reader task fans device messages out
//! to any number of subscribers; writes go straight through a cloned write stream.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use esphome_client::error::{ClientError, ConnectionError, ProtocolError};
use esphome_client::types as pb;
use esphome_client::types::EspHomeMessage as Msg;
use esphome_client::{API_VERSION, EspHomeClient, EspHomeClientWriteStream};
use tokio::sync::broadcast;

pub use pb::EspHomeMessage;

pub struct Client {
    writer: EspHomeClientWriteStream,
    /// weak, so the channel closes as soon as the reader task dies
    events: broadcast::WeakSender<Msg>,
    closed: Closed,
    pub hello: pb::HelloResponse,
}

/// One entity as announced by ListEntities.
#[derive(Debug, Clone)]
pub struct Entity {
    pub kind: &'static str,
    pub key: u32,
    pub object_id: String,
    pub name: String,
    /// options / effects / event types, if any
    pub choices: Vec<String>,
}

impl Client {
    pub async fn connect(addr: &str, key: Option<&str>, client_info: &str) -> Result<Self> {
        let mut b = EspHomeClient::builder()
            .address(addr)
            .timeout(Duration::from_secs(10))
            // we do Hello ourselves: the built-in setup throws the HelloResponse away
            .without_connection_setup();
        if let Some(k) = key {
            b = b.key(k);
        }
        let mut client = b.connect().await.map_err(explain)?;
        let writer = client.write_stream();

        let (events, mut first) = broadcast::channel::<Msg>(1024);
        let weak = events.downgrade();
        let closed = Closed::default();
        let closed_w = closed.clone();
        let auto = client.write_stream();
        tokio::spawn(async move {
            loop {
                // pings are answered inside esphome-client
                match client.try_read().await {
                    Ok(msg) => {
                        if let Msg::DisconnectRequest(_) = msg {
                            let _ = auto.try_write(pb::DisconnectResponse {}).await;
                        }
                        let _ = events.send(msg);
                    }
                    // message type newer than the crate's proto set: frame already consumed, skip it
                    Err(ClientError::Protocol(ProtocolError::ValidationFailed { .. })) => continue,
                    Err(e) => {
                        *closed_w.0.lock().unwrap() = Some(format!("{:#}", explain(e)));
                        break; // drops `events`: every receiver now sees Closed
                    }
                }
            }
        });

        writer
            .try_write(pb::HelloRequest {
                client_info: client_info.into(),
                api_version_major: API_VERSION.0,
                api_version_minor: API_VERSION.1,
            })
            .await
            .map_err(explain)?;
        let hello = wait_for(&mut first, &closed, Duration::from_secs(5), |m| match m {
            Msg::HelloResponse(h) => Some(h.clone()),
            _ => None,
        })
        .await
        .map_err(|e| match key {
            // an encrypted device just drops a plaintext hello
            None => e.context(
                "HAVOICE_KEY is not set and the device requires encryption - \
                 copy .env.example to .env and fill in the key (README: \"Getting the API key\")",
            ),
            Some(_) => e.context("no HelloResponse from device"),
        })?;

        Ok(Self {
            writer,
            events: weak,
            closed,
            hello,
        })
    }

    /// Receiver of every message from the device. Yields `Closed` once the connection drops.
    pub fn subscribe(&self) -> broadcast::Receiver<Msg> {
        match self.events.upgrade() {
            Some(s) => s.subscribe(),
            None => broadcast::channel(1).1, // sender already gone -> immediately Closed
        }
    }

    /// Why the connection closed, if it has.
    pub fn close_reason(&self) -> String {
        self.closed.reason()
    }

    pub async fn send(&self, msg: impl Into<Msg> + std::fmt::Debug) -> Result<()> {
        self.writer
            .try_write(msg)
            .await
            .map_err(|e| anyhow!("send failed ({e}); connection: {}", self.close_reason()))
    }

    pub async fn device_info(&self) -> Result<pb::DeviceInfoResponse> {
        let mut rx = self.subscribe();
        self.send(pb::DeviceInfoRequest {}).await?;
        wait_for(&mut rx, &self.closed, Duration::from_secs(5), |m| match m {
            Msg::DeviceInfoResponse(d) => Some(d.clone()),
            _ => None,
        })
        .await
    }

    pub async fn list_entities(&self) -> Result<Vec<Entity>> {
        let mut rx = self.subscribe();
        self.send(pb::ListEntitiesRequest {}).await?;
        let mut out = Vec::new();
        wait_for(&mut rx, &self.closed, Duration::from_secs(10), |m| {
            if let Some(e) = entity_of(m) {
                out.push(e);
            }
            matches!(m, Msg::ListEntitiesDoneResponse(_)).then_some(())
        })
        .await?;
        Ok(out)
    }

    /// Find one entity of `kind` whose object_id contains every needle.
    pub async fn find(&self, kind: &str, needles: &[&str]) -> Result<Entity> {
        let all = self.list_entities().await?;
        all.iter()
            .find(|e| e.kind == kind && needles.iter().all(|n| e.object_id.contains(n)))
            .cloned()
            .ok_or_else(|| {
                let have: Vec<_> = all
                    .iter()
                    .filter(|e| e.kind == kind)
                    .map(|e| &e.object_id)
                    .collect();
                anyhow!("no {kind} matching {needles:?}; available: {have:?}")
            })
    }

    pub async fn subscribe_states(&self) -> Result<()> {
        self.send(pb::SubscribeStatesRequest {}).await
    }

    pub async fn subscribe_logs(&self, level: pb::LogLevel) -> Result<()> {
        self.send(pb::SubscribeLogsRequest {
            level: level as i32,
            dump_config: false,
        })
        .await
    }

    /// Store a new Noise key (base64) on the device - what Home Assistant does when it
    /// adopts a device. Works on a key-less device over plaintext, or on a paired one over
    /// an encrypted connection (key rotation). An empty key removes encryption.
    /// The device then drops every API client; reconnect with the new key.
    pub async fn set_noise_key(&self, key_b64: &str) -> Result<()> {
        let mut rx = self.subscribe();
        self.send(pb::NoiseEncryptionSetKeyRequest {
            // the device base64-decodes this itself
            key: key_b64.as_bytes().to_vec(),
        })
        .await?;
        let ok = wait_for(&mut rx, &self.closed, Duration::from_secs(5), |m| match m {
            Msg::NoiseEncryptionSetKeyResponse(r) => Some(r.success),
            _ => None,
        })
        .await?;
        anyhow::ensure!(ok, "device refused the new key");
        Ok(())
    }

    pub async fn disconnect(&self) {
        let req = pb::DisconnectRequest {
            reason: pb::DisconnectReason::Unspecified as i32,
        };
        if self.send(req).await.is_ok() {
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

/// A fresh random 32-byte Noise PSK, base64 - the same format Home Assistant generates.
pub fn generate_key() -> Result<String> {
    use base64::Engine as _;
    let mut k = [0u8; 32];
    getrandom::fill(&mut k).map_err(|e| anyhow!("no randomness: {e}"))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(k))
}

/// Turn the crate's errors into hints a human can act on.
fn explain(e: ClientError) -> anyhow::Error {
    let hint = match &e {
        ClientError::Connection(ConnectionError::NoiseHandshake { .. }) => {
            "device rejected the handshake - wrong HAVOICE_KEY?"
        }
        ClientError::Protocol(ProtocolError::UnexpectedEncryption) => {
            "device requires encryption - set HAVOICE_KEY"
        }
        ClientError::Protocol(ProtocolError::UnexpectedPlain) => {
            "device speaks plaintext - unset HAVOICE_KEY"
        }
        _ => return anyhow!(e),
    };
    anyhow!(e).context(hint)
}

#[derive(Clone, Default)]
struct Closed(Arc<Mutex<Option<String>>>);

impl Closed {
    fn reason(&self) -> String {
        self.0
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| "closed".into())
    }
}

async fn wait_for<T>(
    rx: &mut broadcast::Receiver<Msg>,
    closed: &Closed,
    timeout: Duration,
    mut pick: impl FnMut(&Msg) -> Option<T>,
) -> Result<T> {
    let fut = async {
        loop {
            match rx.recv().await {
                Ok(m) => {
                    if let Some(v) = pick(&m) {
                        return Ok(v);
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => bail!("lost {n} messages"),
                Err(_) => bail!("connection closed: {}", closed.reason()),
            }
        }
    };
    tokio::time::timeout(timeout, fut)
        .await
        .map_err(|_| anyhow!("timed out waiting for device"))?
}

/// ESPHome object_id: lower snake case of the name (newer firmware leaves the field empty).
fn object_id(given: &str, name: &str) -> String {
    if !given.is_empty() {
        return given.to_string();
    }
    name.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

macro_rules! entity {
    ($m:expr, $kind:literal, $choices:expr) => {
        Some(Entity {
            kind: $kind,
            key: $m.key,
            object_id: object_id(&$m.object_id, &$m.name),
            name: $m.name.clone(),
            choices: $choices,
        })
    };
}

fn entity_of(msg: &Msg) -> Option<Entity> {
    match msg {
        Msg::ListEntitiesBinarySensorResponse(m) => entity!(m, "binary_sensor", vec![]),
        Msg::ListEntitiesSensorResponse(m) => entity!(m, "sensor", vec![]),
        Msg::ListEntitiesTextSensorResponse(m) => entity!(m, "text_sensor", vec![]),
        Msg::ListEntitiesSwitchResponse(m) => entity!(m, "switch", vec![]),
        Msg::ListEntitiesLightResponse(m) => entity!(m, "light", m.effects.clone()),
        Msg::ListEntitiesSelectResponse(m) => entity!(m, "select", m.options.clone()),
        Msg::ListEntitiesNumberResponse(m) => entity!(m, "number", vec![]),
        Msg::ListEntitiesButtonResponse(m) => entity!(m, "button", vec![]),
        Msg::ListEntitiesMediaPlayerResponse(m) => entity!(m, "media_player", vec![]),
        Msg::ListEntitiesEventResponse(m) => entity!(m, "event", m.event_types.clone()),
        Msg::ListEntitiesUpdateResponse(m) => entity!(m, "update", vec![]),
        Msg::ListEntitiesTextResponse(m) => entity!(m, "text", vec![]),
        _ => None,
    }
}

/// Key of a state message, so `watch` can name it.
pub fn state_key(msg: &Msg) -> Option<u32> {
    Some(match msg {
        Msg::BinarySensorStateResponse(m) => m.key,
        Msg::SensorStateResponse(m) => m.key,
        Msg::TextSensorStateResponse(m) => m.key,
        Msg::SwitchStateResponse(m) => m.key,
        Msg::LightStateResponse(m) => m.key,
        Msg::SelectStateResponse(m) => m.key,
        Msg::NumberStateResponse(m) => m.key,
        Msg::MediaPlayerStateResponse(m) => m.key,
        Msg::EventResponse(m) => m.key,
        Msg::UpdateStateResponse(m) => m.key,
        Msg::TextStateResponse(m) => m.key,
        _ => return None,
    })
}

pub fn names_by_key(entities: &[Entity]) -> HashMap<u32, String> {
    entities
        .iter()
        .map(|e| (e.key, format!("{}.{}", e.kind, e.object_id)))
        .collect()
}
