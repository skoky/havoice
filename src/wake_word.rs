//! Wake word models (microWakeWord) for the Voice PE.
//!
//! * [`verify`] checks a `.tflite` + manifest the same way the device firmware does when it
//!   loads a model (ESPHome 2026.6 `micro_wake_word`): flatbuffer schema, the 20 operators
//!   its interpreter registers, input `int8 [1, stride, 40]`, output `uint8 [1, 1]`.
//! * [`firmware_yaml`] writes an ESPHome config: the stock Voice PE firmware plus extra
//!   models. The firmware has no way to receive a model at runtime, so a new model means
//!   building and flashing that config (`examples/wake_word.rs install --flash`).
//! * [`Detections`] follows the device log and yields every wake word detection.
//!
//! ```no_run
//! # async fn demo() -> anyhow::Result<()> {
//! use havoice::wake_word::{self, Detections, Manifest};
//!
//! // check a model before building it into firmware
//! let manifest = Manifest::parse(&std::fs::read("alexa.json")?)?;
//! let report = wake_word::verify(&manifest, &std::fs::read("alexa.tflite")?, None);
//! for c in &report.checks {
//!     println!("{} {}", if c.ok { "ok  " } else { "FAIL" }, c.what);
//! }
//!
//! // watch the device hear its wake words
//! let pe = havoice::VoicePe::from_env().await?;
//! for m in wake_word::device_models(&pe).await? {
//!     println!("model: {}", m.wake_word);
//! }
//! let mut detections = Detections::start(&pe).await?;
//! while let Some(d) = detections.next().await {
//!     println!("heard {} ({:?})", d.wake_word, d.average);
//! }
//! # Ok(()) }
//! ```

use std::collections::BTreeSet;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use esphome_client::types::{self as pb, EspHomeMessage as Msg};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::VoicePe;

// ------------------------------------------------------------------- manifest

/// microWakeWord manifest (`*.json` next to the `.tflite`), version 2.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    /// model family; only `micro` (microWakeWord) runs on the device
    #[serde(rename = "type")]
    pub kind: String,
    /// the phrase, e.g. `Hey Jarvis` - also what detections report
    pub wake_word: String,
    /// who trained it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// where it comes from
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub website: Option<String>,
    /// file name of the `.tflite`, relative to the manifest
    pub model: String,
    /// language codes the model was trained on, e.g. `["en"]`
    #[serde(default)]
    pub trained_languages: Vec<String>,
    /// manifest format version (2)
    pub version: u32,
    /// runtime settings for the firmware
    pub micro: MicroSettings,
}

/// The `micro` block of a [`Manifest`]: how the firmware runs the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MicroSettings {
    /// detection threshold 0-1; higher = fewer false wakes, more misses
    pub probability_cutoff: f32,
    /// audio feature step in ms (10 for every official model)
    pub feature_step_size: u32,
    /// how many consecutive probabilities are averaged before comparing to the cutoff
    pub sliding_window_size: u32,
    /// working memory the model needs, in bytes
    pub tensor_arena_size: u32,
    /// oldest ESPHome that can run it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_esphome_version: Option<String>,
}

impl Manifest {
    /// Parse a manifest `.json`.
    pub fn parse(json: &[u8]) -> Result<Self> {
        serde_json::from_slice(json).context("not a microWakeWord manifest")
    }

    /// A manifest for a bare `.tflite` (no JSON came with it). The defaults are those of
    /// the official v2 models; the tensor arena is a safe upper bound for them.
    pub fn for_custom(wake_word: &str, model_file: &str, language: Option<&str>) -> Self {
        Self {
            kind: "micro".into(),
            wake_word: wake_word.into(),
            author: None,
            website: None,
            model: model_file.into(),
            trained_languages: language.map(|l| vec![l.into()]).unwrap_or_default(),
            version: 2,
            micro: MicroSettings {
                probability_cutoff: 0.97,
                feature_step_size: 10,
                sliding_window_size: 5,
                tensor_arena_size: 30_000,
                minimum_esphome_version: Some("2024.7.0".into()),
            },
        }
    }
}

// ------------------------------------------------------------------- tflite

/// What the device cares about in a `.tflite`.
#[derive(Debug, Clone)]
pub struct TfliteInfo {
    /// TFLite schema version (the device needs 3)
    pub schema_version: u32,
    /// builtin operator codes, by name where known
    pub operators: Vec<String>,
    /// operators the device's interpreter doesn't register (must be empty)
    pub unsupported: Vec<String>,
    /// main input shape; the device needs `[1, stride, 40]`
    pub input_shape: Vec<i32>,
    /// main input type; the device needs `int8`
    pub input_type: &'static str,
    /// output shape; the device needs `[1, 1]`
    pub output_shape: Vec<i32>,
    /// output type; the device needs `uint8`
    pub output_type: &'static str,
    /// number of subgraphs (streaming models have an extra init graph)
    pub subgraphs: usize,
}

/// Operators the firmware's interpreter registers (`streaming_model.cpp`,
/// `MicroMutableOpResolver<20>`). A model using anything else fails to load on the device.
pub const SUPPORTED_OPS: [(i32, &str); 20] = [
    (0, "ADD"),
    (1, "AVERAGE_POOL_2D"),
    (2, "CONCATENATION"),
    (3, "CONV_2D"),
    (4, "DEPTHWISE_CONV_2D"),
    (9, "FULLY_CONNECTED"),
    (14, "LOGISTIC"),
    (17, "MAX_POOL_2D"),
    (18, "MUL"),
    (22, "RESHAPE"),
    (34, "PAD"),
    (40, "MEAN"),
    (45, "STRIDED_SLICE"),
    (83, "PACK"),
    (102, "SPLIT_V"),
    (114, "QUANTIZE"),
    (129, "CALL_ONCE"),
    (142, "VAR_HANDLE"),
    (143, "READ_VARIABLE"),
    (144, "ASSIGN_VARIABLE"),
];

/// Names for operators that commonly show up in models converted from other frameworks.
const OTHER_OPS: [(i32, &str); 18] = [
    (6, "DEQUANTIZE"),
    (19, "RELU"),
    (21, "RELU6"),
    (25, "SOFTMAX"),
    (28, "TANH"),
    (32, "CUSTOM"),
    (36, "GATHER"),
    (39, "TRANSPOSE"),
    (41, "SUB"),
    (42, "DIV"),
    (43, "SQUEEZE"),
    (75, "SQRT"),
    (76, "RSQRT"),
    (78, "POW"),
    (98, "LEAKY_RELU"),
    (99, "SQUARED_DIFFERENCE"),
    (126, "BATCH_MATMUL"),
    (44, "UNIDIRECTIONAL_SEQUENCE_LSTM"),
];

fn op_name(code: i32) -> String {
    SUPPORTED_OPS
        .iter()
        .chain(OTHER_OPS.iter())
        .find(|(c, _)| *c == code)
        .map(|(_, n)| n.to_string())
        .unwrap_or_else(|| format!("op#{code}"))
}

fn tensor_type(t: i8) -> &'static str {
    match t {
        0 => "float32",
        1 => "float16",
        2 => "int32",
        3 => "uint8",
        4 => "int64",
        6 => "bool",
        7 => "int16",
        9 => "int8",
        _ => "other",
    }
}

/// Minimal read-only flatbuffer access - just enough of the TFLite schema.
struct Fb<'a>(&'a [u8]);

impl<'a> Fb<'a> {
    fn u32(&self, at: usize) -> Result<u32> {
        let b = self
            .0
            .get(at..at + 4)
            .ok_or_else(|| anyhow!("truncated file"))?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn i32(&self, at: usize) -> Result<i32> {
        Ok(self.u32(at)? as i32)
    }
    fn u16(&self, at: usize) -> Result<u16> {
        let b = self
            .0
            .get(at..at + 2)
            .ok_or_else(|| anyhow!("truncated file"))?;
        Ok(u16::from_le_bytes(b.try_into().unwrap()))
    }
    fn i8(&self, at: usize) -> Result<i8> {
        Ok(*self.0.get(at).ok_or_else(|| anyhow!("truncated file"))? as i8)
    }
    /// Absolute position of field `n` of the table at `table`, if present.
    fn field(&self, table: usize, n: usize) -> Result<Option<usize>> {
        let vtable = (table as i64 - self.i32(table)? as i64) as usize;
        let vt_len = self.u16(vtable)? as usize;
        let slot = 4 + 2 * n;
        if slot + 2 > vt_len {
            return Ok(None);
        }
        let off = self.u16(vtable + slot)? as usize;
        Ok((off != 0).then_some(table + off))
    }
    /// Follow an offset stored at `at` (to a table, vector or string).
    fn deref(&self, at: usize) -> Result<usize> {
        Ok(at + self.u32(at)? as usize)
    }
    /// (start, len) of the vector referenced by field `n`.
    fn vector(&self, table: usize, n: usize) -> Result<Option<(usize, usize)>> {
        match self.field(table, n)? {
            None => Ok(None),
            Some(f) => {
                let v = self.deref(f)?;
                Ok(Some((v + 4, self.u32(v)? as usize)))
            }
        }
    }
    fn tables(&self, table: usize, n: usize) -> Result<Vec<usize>> {
        let Some((start, len)) = self.vector(table, n)? else {
            return Ok(vec![]);
        };
        (0..len).map(|i| self.deref(start + 4 * i)).collect()
    }
    fn ints(&self, table: usize, n: usize) -> Result<Vec<i32>> {
        let Some((start, len)) = self.vector(table, n)? else {
            return Ok(vec![]);
        };
        (0..len).map(|i| self.i32(start + 4 * i)).collect()
    }
}

/// Read the parts of a `.tflite` the device checks.
pub fn inspect_tflite(bytes: &[u8]) -> Result<TfliteInfo> {
    ensure!(
        bytes.len() > 8 && &bytes[4..8] == b"TFL3",
        "not a TensorFlow Lite file (no TFL3 identifier)"
    );
    let fb = Fb(bytes);
    let model = fb.deref(0)?;
    // Model: 0 version, 1 operator_codes, 2 subgraphs
    let schema_version = match fb.field(model, 0)? {
        Some(f) => fb.u32(f)?,
        None => 0,
    };
    let mut codes = Vec::new();
    for oc in fb.tables(model, 1)? {
        // OperatorCode: 0 deprecated_builtin_code (int8), 3 builtin_code (int32)
        let deprecated = match fb.field(oc, 0)? {
            Some(f) => fb.i8(f)? as i32,
            None => 0,
        };
        let builtin = match fb.field(oc, 3)? {
            Some(f) => fb.i32(f)?,
            None => 0,
        };
        codes.push(deprecated.max(builtin));
    }
    let subgraphs = fb.tables(model, 2)?;
    let main = *subgraphs.first().context("model has no subgraph")?;
    // SubGraph: 0 tensors, 1 inputs, 2 outputs
    let tensors = fb.tables(main, 0)?;
    let tensor = |idx: i32| -> Result<(Vec<i32>, &'static str)> {
        let t = *tensors
            .get(idx as usize)
            .ok_or_else(|| anyhow!("bad tensor index {idx}"))?;
        // Tensor: 0 shape, 1 type
        let ty = match fb.field(t, 1)? {
            Some(f) => fb.i8(f)?,
            None => 0,
        };
        Ok((fb.ints(t, 0)?, tensor_type(ty)))
    };
    let inputs = fb.ints(main, 1)?;
    let outputs = fb.ints(main, 2)?;
    ensure!(
        inputs.len() == 1 && outputs.len() == 1,
        "model must have exactly one input and one output (has {} and {})",
        inputs.len(),
        outputs.len()
    );
    let (input_shape, input_type) = tensor(inputs[0])?;
    let (output_shape, output_type) = tensor(outputs[0])?;

    let supported: BTreeSet<i32> = SUPPORTED_OPS.iter().map(|(c, _)| *c).collect();
    let unique: BTreeSet<i32> = codes.iter().copied().collect();
    Ok(TfliteInfo {
        schema_version,
        operators: unique.iter().map(|c| op_name(*c)).collect(),
        unsupported: unique
            .iter()
            .filter(|c| !supported.contains(c))
            .map(|c| op_name(*c))
            .collect(),
        input_shape,
        input_type,
        output_shape,
        output_type,
        subgraphs: subgraphs.len(),
    })
}

// ------------------------------------------------------------------- verify

/// One line of a verification report.
#[derive(Debug, Clone)]
pub struct Check {
    /// the check passed
    pub ok: bool,
    /// a warning doesn't block installing
    pub warning: bool,
    /// what was checked, with the value found
    pub what: String,
}

/// Result of [`verify`].
#[derive(Debug, Clone)]
pub struct Report {
    /// every check, in order
    pub checks: Vec<Check>,
    /// what was read from the `.tflite`, if it could be read at all
    pub tflite: Option<TfliteInfo>,
    /// SHA-256 of the `.tflite`, hex
    pub sha256: String,
    /// size of the `.tflite` in bytes
    pub size: usize,
}

impl Report {
    /// No failed check (warnings allowed).
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.ok || c.warning)
    }
}

/// The biggest tensor arena a model may ask for on the Voice PE (8 MB PSRAM, but the
/// official models need 22-45 kB; anything far beyond that is almost certainly wrong).
const MAX_ARENA: u32 = 200_000;

/// Check a model + manifest before it goes into firmware. `esphome_version` (e.g.
/// `2026.10.2`) enables the minimum-version check.
pub fn verify(manifest: &Manifest, tflite: &[u8], esphome_version: Option<&str>) -> Report {
    let mut checks = Vec::new();
    let mut check = |ok: bool, what: String| {
        checks.push(Check {
            ok,
            warning: false,
            what,
        })
    };
    let m = &manifest.micro;
    check(
        manifest.kind == "micro",
        format!(
            "manifest type is \"micro\" (got {:?}) - only microWakeWord models run on the device",
            manifest.kind
        ),
    );
    check(
        manifest.version == 2,
        format!("manifest version 2 (got {})", manifest.version),
    );
    check(
        !manifest.wake_word.trim().is_empty(),
        "wake word name is set".into(),
    );
    check(
        m.probability_cutoff > 0.0 && m.probability_cutoff <= 1.0,
        format!("probability cutoff in (0, 1]: {}", m.probability_cutoff),
    );
    check(
        (1..=100).contains(&m.sliding_window_size),
        format!("sliding window 1-100: {}", m.sliding_window_size),
    );
    check(
        m.feature_step_size <= 30,
        format!("feature step 0-30 ms: {}", m.feature_step_size),
    );
    check(
        m.tensor_arena_size > 0 && m.tensor_arena_size <= MAX_ARENA,
        format!("tensor arena 1-{MAX_ARENA} bytes: {}", m.tensor_arena_size),
    );

    let tflite_info = match inspect_tflite(tflite) {
        Ok(info) => {
            check(true, "TensorFlow Lite flatbuffer (TFL3)".into());
            check(
                info.schema_version == 3,
                format!("TFLite schema version 3 (got {})", info.schema_version),
            );
            check(
                info.unsupported.is_empty(),
                if info.unsupported.is_empty() {
                    format!(
                        "all {} operators are supported by the device",
                        info.operators.len()
                    )
                } else {
                    format!(
                        "operators the device can't run: {}",
                        info.unsupported.join(", ")
                    )
                },
            );
            let s = &info.input_shape;
            check(
                s.len() == 3 && s[0] == 1 && s[2] == 40,
                format!("input shape [1, stride, 40] (got {s:?})"),
            );
            check(
                info.input_type == "int8",
                format!("input is int8 (got {})", info.input_type),
            );
            check(
                info.output_shape == [1, 1],
                format!("output shape [1, 1] (got {:?})", info.output_shape),
            );
            check(
                info.output_type == "uint8",
                format!("output is uint8 (got {})", info.output_type),
            );
            Some(info)
        }
        Err(e) => {
            check(false, format!("readable .tflite: {e:#}"));
            None
        }
    };

    let mut report = Report {
        checks,
        tflite: tflite_info,
        sha256: sha256_hex(tflite),
        size: tflite.len(),
    };
    if m.feature_step_size != 10 {
        report.checks.push(Check {
            ok: false,
            warning: true,
            what: format!(
                "feature step {} ms differs from the stock models (10 ms) - ESPHome rejects mixing them",
                m.feature_step_size
            ),
        });
    }
    if let (Some(min), Some(have)) = (&m.minimum_esphome_version, esphome_version) {
        report.checks.push(Check {
            ok: version_le(min, have),
            warning: false,
            what: format!("needs ESPHome {min} or newer (building with {have})"),
        });
    }
    report
}

/// `a <= b` for dotted versions like `2024.7.0`.
fn version_le(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Vec<u32> {
        v.split(['.', '-'])
            .map(|p| p.parse().unwrap_or(0))
            .collect()
    };
    parse(a) <= parse(b)
}

/// SHA-256 of `bytes` as lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ------------------------------------------------------------------- firmware

/// Where the generated firmware comes from and what it keeps.
#[derive(Debug, Clone)]
pub struct Firmware {
    /// device hostname without `.local`, e.g. `home-assistant-voice-0a1b2c`
    pub name: String,
    /// name shown in Home Assistant, e.g. `Home Assistant Voice 0a1b2c`
    pub friendly_name: String,
    /// stock config, from the device's mDNS `package_import_url`
    pub package_url: String,
    /// `(id, manifest path relative to the YAML)` of every extra model
    pub models: Vec<(String, String)>,
}

/// ESPHome YAML: the stock Voice PE firmware plus extra wake word models. Same name and
/// API key (`!secret havoice_api_key`), so Home Assistant and havoice keep working.
/// Each extra model is switched on whenever an API client connects - by default the
/// firmware only enables the first model, and Home Assistant picks the active ones.
pub fn firmware_yaml(fw: &Firmware) -> String {
    let mut y = String::new();
    y.push_str(
        "# Generated by havoice (examples/wake_word.rs install). Stock Voice PE firmware plus\n\
         # extra wake word models. Rebuild with `install` after changing the models/ folder;\n\
         # go back to stock firmware with https://esphome.github.io/home-assistant-voice-pe/\n",
    );
    y.push_str(&format!(
        "substitutions:\n  name: {}\n  friendly_name: {}\n\n",
        fw.name,
        yaml_str(&fw.friendly_name)
    ));
    y.push_str(&format!(
        "packages:\n  voice_pe: {}\n\n",
        yaml_str(&fw.package_url)
    ));
    y.push_str(
        "esphome:\n  name: ${name}\n  name_add_mac_suffix: false\n  friendly_name: ${friendly_name}\n\n",
    );
    y.push_str(
        "# Wi-Fi stays the network saved on the device (set up via Improv / Home Assistant).\n\
         # ESPHome only accepts a config with an SSID or an access point, so this adds a\n\
         # fallback hotspot that starts only if the saved network is unreachable for a minute.\n\
         wifi:\n  ap:\n    password: !secret havoice_ap_password\n\n",
    );
    y.push_str("api:\n  encryption:\n    key: !secret havoice_api_key\n");
    if !fw.models.is_empty() {
        y.push_str("  on_client_connected:\n");
        for (id, _) in &fw.models {
            y.push_str(&format!("    - micro_wake_word.enable_model: {id}\n"));
        }
        y.push_str("\nmicro_wake_word:\n  models:\n");
        for (id, path) in &fw.models {
            y.push_str(&format!(
                "    - model: {}\n      id: {id}\n",
                yaml_str(path)
            ));
        }
    }
    y
}

fn yaml_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// A safe ESPHome id from a wake word: `Hey Home Assistant` -> `hey_home_assistant`.
pub fn model_id(wake_word: &str) -> String {
    let mut id: String = wake_word
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    while id.contains("__") {
        id = id.replace("__", "_");
    }
    let id = id.trim_matches('_').to_string();
    if id.is_empty() || id.starts_with(|c: char| c.is_ascii_digit()) {
        format!("ww_{id}")
    } else {
        id
    }
}

// ------------------------------------------------------------------- device side

/// A wake word model compiled into the running firmware.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceModel {
    /// the phrase, e.g. `Okay Nabu` (`Stop` is the built-in stop-timer model)
    pub wake_word: String,
    /// detection threshold in effect (shifted by the sensitivity setting)
    pub probability_cutoff: Option<f32>,
}

/// Models in the running firmware, read from the device's own config dump (works next to
/// Home Assistant, unlike the voice assistant configuration request).
pub async fn device_models(pe: &VoicePe) -> Result<Vec<DeviceModel>> {
    let mut rx = pe.client().subscribe();
    pe.client()
        .send(pb::SubscribeLogsRequest {
            level: pb::LogLevel::Config as i32,
            dump_config: true,
        })
        .await?;
    let mut models: Vec<DeviceModel> = Vec::new();
    let mut seen_section = false;
    let quiet = Duration::from_millis(1500);
    loop {
        // the dump arrives in a burst; stop once it has gone quiet for a moment
        let wait = if seen_section {
            quiet
        } else {
            Duration::from_secs(8)
        };
        let msg = match tokio::time::timeout(wait, rx.recv()).await {
            Err(_) => break,
            Ok(Err(broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(_)) => bail!("connection closed: {}", pe.client().close_reason()),
            Ok(Ok(m)) => m,
        };
        let Msg::SubscribeLogsResponse(l) = msg else {
            continue;
        };
        let text = strip_ansi(&String::from_utf8_lossy(&l.message));
        if !text.contains("micro_wake_word") {
            continue;
        }
        seen_section = true;
        for line in text.lines() {
            if let Some(name) = line.split("Wake Word: ").nth(1) {
                models.push(DeviceModel {
                    wake_word: name.trim().to_string(),
                    probability_cutoff: None,
                });
            } else if let Some(v) = line.split("Probability cutoff: ").nth(1)
                && let Some(last) = models.last_mut()
                && last.probability_cutoff.is_none()
            {
                last.probability_cutoff = v.trim().parse().ok();
            }
        }
    }
    ensure!(
        seen_section,
        "the device logged no micro_wake_word config - is wake word detection in this firmware?"
    );
    Ok(models)
}

/// One detection reported by the device.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// the phrase the model is for
    pub wake_word: String,
    /// sliding-window average probability, 0-1 (`None` for a VAD rejection)
    pub average: Option<f32>,
    /// peak probability in the window, 0-1
    pub max: Option<f32>,
    /// the model fired but voice activity detection said nobody was speaking
    pub vad_rejected: bool,
}

/// Parse a `micro_wake_word` log line.
pub fn parse_detection(line: &str) -> Option<Detection> {
    let line = strip_ansi(line);
    if let Some(rest) = line.split("Detected '").nth(1) {
        let (word, rest) = rest.split_once('\'')?;
        let number = |key: &str| -> Option<f32> {
            let v = rest.split(key).nth(1)?;
            v.trim()
                .split(|c: char| !(c.is_ascii_digit() || c == '.'))
                .next()?
                .parse()
                .ok()
        };
        return Some(Detection {
            wake_word: word.to_string(),
            average: number("average probability is"),
            max: number("max probability is"),
            vad_rejected: false,
        });
    }
    let rest = line.split("Wake word model predicts '").nth(1)?;
    let (word, _) = rest.split_once('\'')?;
    Some(Detection {
        wake_word: word.to_string(),
        average: None,
        max: None,
        vad_rejected: true,
    })
}

/// Live wake word detections, from the device log (DEBUG level, which the stock firmware
/// uses). Works alongside Home Assistant, which still runs the voice pipeline.
pub struct Detections {
    rx: broadcast::Receiver<Msg>,
}

impl Detections {
    /// Start following the device log on `pe`'s connection.
    pub async fn start(pe: &VoicePe) -> Result<Self> {
        let rx = pe.client().subscribe();
        pe.client()
            .send(pb::SubscribeLogsRequest {
                level: pb::LogLevel::Debug as i32,
                dump_config: false,
            })
            .await?;
        Ok(Self { rx })
    }

    /// Next detection; `None` once the connection closes.
    pub async fn next(&mut self) -> Option<Detection> {
        loop {
            match self.rx.recv().await {
                Ok(Msg::SubscribeLogsResponse(l)) => {
                    if let Some(d) = parse_detection(&String::from_utf8_lossy(&l.message)) {
                        return Some(d);
                    }
                }
                Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(_) => return None,
            }
        }
    }
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            for d in chars.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_lines() {
        let d = parse_detection(
            "\x1b[0;36m[D][micro_wake_word:334]: Detected 'Okay Nabu' with sliding average probability is 0.96 and max probability is 1.00\x1b[0m",
        )
        .unwrap();
        assert_eq!(d.wake_word, "Okay Nabu");
        assert_eq!(d.average, Some(0.96));
        assert_eq!(d.max, Some(1.0));
        assert!(!d.vad_rejected);
        let v = parse_detection(
            "[D][micro_wake_word:330]: Wake word model predicts 'Hey Jarvis', but VAD model doesn't.",
        )
        .unwrap();
        assert!(v.vad_rejected && v.wake_word == "Hey Jarvis");
        assert_eq!(parse_detection("[D][wifi]: something else"), None);
    }

    #[test]
    fn manifest_and_ids() {
        let m = Manifest::parse(
            br#"{"type":"micro","wake_word":"Hey Home Assistant","author":"Michael Hansen",
                "model":"hey_home_assistant.tflite","trained_languages":["en"],"version":2,
                "micro":{"probability_cutoff":0.97,"feature_step_size":10,"sliding_window_size":5,
                "tensor_arena_size":30000,"minimum_esphome_version":"2024.7.0"}}"#,
        )
        .unwrap();
        assert_eq!(model_id(&m.wake_word), "hey_home_assistant");
        assert_eq!(model_id("Ahoj, Hugo!"), "ahoj_hugo");
        assert_eq!(model_id("3 2 1"), "ww_3_2_1");
        assert!(version_le("2024.7.0", "2026.10.2"));
        assert!(!version_le("2026.11.0", "2026.10.2"));
        // a manifest alone fails on the missing/garbage model
        let r = verify(&m, b"not a model", Some("2026.10.2"));
        assert!(!r.passed());
    }

    #[test]
    fn yaml_keeps_stock_and_adds_models() {
        let y = firmware_yaml(&Firmware {
            name: "home-assistant-voice-0a1b2c".into(),
            friendly_name: "Home Assistant Voice 0a1b2c".into(),
            package_url: "github://esphome/home-assistant-voice-pe/home-assistant-voice.yaml"
                .into(),
            models: vec![("alexa".into(), "models/alexa.json".into())],
        });
        assert!(y.contains("voice_pe: \"github://esphome/home-assistant-voice-pe/"));
        assert!(y.contains("key: !secret havoice_api_key"));
        assert!(y.contains("wifi:\n  ap:\n    password: !secret havoice_ap_password"));
        assert!(y.contains("- micro_wake_word.enable_model: alexa"));
        assert!(y.contains("- model: \"models/alexa.json\"\n      id: alexa"));
    }

    /// Reads a real model when one is lying around (HAVOICE_TEST_TFLITE=path).
    #[test]
    fn real_model_if_available() {
        let Ok(path) = std::env::var("HAVOICE_TEST_TFLITE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let info = inspect_tflite(&bytes).unwrap();
        println!("{info:#?}");
        assert!(info.unsupported.is_empty(), "{:?}", info.unsupported);
        assert_eq!(info.input_type, "int8");
        assert_eq!(info.output_type, "uint8");
    }
}
