//! Add wake word models to the Voice PE and watch it detect them.
//!
//!     cargo run --example wake_word -- list                       # on the device + online catalogue
//!     cargo run --example wake_word -- verify alexa               # download + check, change nothing
//!     cargo run --example wake_word -- verify my.tflite --wake-word "Ahoj Hugo" --language cs
//!     cargo run --example wake_word -- install alexa              # store + write firmware YAML
//!     cargo run --example wake_word -- install alexa --flash      # ... then build and flash over Wi-Fi
//!     cargo run --example wake_word -- remove alexa [--flash]
//!     cargo run --example wake_word -- build [--flash]             # rebuild from firmware/models/
//!     cargo run --example wake_word -- convert my.onnx --wake-word "Ahoj Hugo"   # ONNX -> .tflite
//!     cargo run --example wake_word -- listen                     # print every detection
//!
//! A model source is a catalogue name (`alexa`, `hey_home_assistant`, ...), an http(s) URL of a
//! manifest `.json` or a `.tflite`, or a local `.json` / `.tflite`. A bare `.tflite` needs
//! `--wake-word` (and optionally `--language`, `--cutoff`, `--window`, `--arena`).
//!
//! Every model is verified the way the device checks it when loading: TFLite schema, only the
//! 20 operators the firmware's interpreter has, input int8 [1, stride, 40], output uint8 [1, 1],
//! sane manifest values, ESPHome version. Nothing that fails is stored.
//!
//! The firmware can't receive a model at runtime (ESPHome compiles models in), so `install`
//! keeps verified models in `firmware/models/` and writes `firmware/voice-pe-wake-words.yaml`:
//! the stock Voice PE firmware + those models, same device name and API key (HAVOICE_KEY).
//! `--flash` builds it with the ESPHome CLI (`esphome` on PATH, `HAVOICE_ESPHOME`, or
//! `.venv-esphome/bin/esphome`; the first build downloads ~1 GB of toolchain and takes
//! 10-20 minutes) and uploads it over Wi-Fi. Back to stock firmware:
//! https://esphome.github.io/home-assistant-voice-pe/
//!
//! Extra models switch on whenever an API client connects; in Home Assistant you can pick the
//! active wake words on the device page as usual.
//!
//! `convert` takes an ONNX model (e.g. from openWakeWord) and tries to turn it into a
//! microWakeWord `.tflite` with [onnx2tf](https://github.com/PINTO0309/onnx2tf) - needs
//! `pip install onnx2tf tensorflow`. The device only runs streaming models trained on
//! melspectrogram features, so a model that wasn't trained that way usually can't be converted
//! into something that detects; `convert` reports how far it gets and checks every candidate
//! it produces with the same rules as `verify`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand};
use havoice::device::load_env;
use havoice::wake_word::{self, Detections, Firmware, Manifest, Report};
use havoice::{VoicePe, discover};
use havoice::{onnx, wake_word::Check};

const CATALOGUE: &str =
    "https://raw.githubusercontent.com/esphome/micro-wake-word-models/main/models/v2";
const CATALOGUE_DIRS: &str =
    "https://api.github.com/repos/esphome/micro-wake-word-models/contents/models/v2";
const STOCK_PACKAGE: &str = "github://esphome/home-assistant-voice-pe/home-assistant-voice.yaml";
const USER_AGENT: &str = "havoice-example/0.1";
const YAML: &str = "voice-pe-wake-words.yaml";

/// Add wake word models to the Voice PE and watch it detect them.
#[derive(Parser)]
#[command(
    after_help = "Model source: a catalogue name (see `list`), an http(s) URL of a .json \
manifest or a .tflite, or a local .json / .tflite file."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Models in the running firmware, stored for the next build, and in the online catalogue
    List,
    /// Download (if needed) and check models; changes nothing
    Verify {
        #[arg(required = true)]
        sources: Vec<String>,
        #[command(flatten)]
        custom: Custom,
    },
    /// Verify and store models, write the firmware YAML; with --flash build and upload it
    Install {
        #[arg(required = true)]
        sources: Vec<String>,
        #[command(flatten)]
        custom: Custom,
        /// Build with the ESPHome CLI and upload over Wi-Fi
        #[arg(long)]
        flash: bool,
    },
    /// Remove stored models and rewrite the firmware YAML; with --flash build and upload it
    Remove {
        #[arg(required = true)]
        ids: Vec<String>,
        /// Build with the ESPHome CLI and upload over Wi-Fi
        #[arg(long)]
        flash: bool,
    },
    /// Rewrite the firmware YAML from the stored models; with --flash build and upload it
    Build {
        /// Build with the ESPHome CLI and upload over Wi-Fi
        #[arg(long)]
        flash: bool,
    },
    /// Try to convert an ONNX model into a microWakeWord .tflite (needs pip install onnx2tf tensorflow)
    Convert {
        /// Path to the .onnx file
        path: PathBuf,
        #[command(flatten)]
        custom: Custom,
    },
    /// Print every wake word detection from the device log
    Listen,
}

/// Settings for a bare `.tflite`, or overrides on top of its manifest.
#[derive(Args, Default)]
struct Custom {
    /// The phrase, e.g. "Ahoj Hugo" (required for a .tflite without manifest)
    #[arg(long)]
    wake_word: Option<String>,
    /// Trained language, e.g. cs
    #[arg(long)]
    language: Option<String>,
    /// Probability cutoff 0-1 (higher = fewer false wakes)
    #[arg(long)]
    cutoff: Option<f32>,
    /// Sliding window size (feature frames averaged)
    #[arg(long)]
    window: Option<u32>,
    /// Tensor arena size in bytes
    #[arg(long)]
    arena: Option<u32>,
}

/// A model in memory, before it's stored.
struct Model {
    manifest: Manifest,
    tflite: Vec<u8>,
    source: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    load_env();
    match Cli::parse().cmd {
        Cmd::List => list().await,
        Cmd::Verify { sources, custom } => {
            let esphome = esphome_version().await.ok();
            let mut all_ok = true;
            for s in &sources {
                let m = load(s, &custom).await?;
                all_ok &= print_report(&m, &check(&m, esphome.as_deref()));
            }
            ensure!(all_ok, "verification failed");
            Ok(())
        }
        Cmd::Install {
            sources,
            custom,
            flash,
        } => {
            let esphome = esphome_version().await.ok();
            for s in &sources {
                let m = load(s, &custom).await?;
                ensure!(
                    print_report(&m, &check(&m, esphome.as_deref())),
                    "{s}: verification failed - not stored"
                );
                let id = store(&m)?;
                println!("stored as firmware/models/{id}.json + {id}.tflite\n");
            }
            build(flash).await
        }
        Cmd::Remove { ids, flash } => {
            for id in &ids {
                let id = wake_word::model_id(id);
                let mut gone = false;
                for ext in ["json", "tflite"] {
                    let p = models_dir().join(format!("{id}.{ext}"));
                    if p.exists() {
                        std::fs::remove_file(&p)?;
                        gone = true;
                    }
                }
                ensure!(gone, "no stored model {id} (see firmware/models/)");
                println!("removed {id}");
            }
            build(flash).await
        }
        Cmd::Build { flash } => build(flash).await,
        Cmd::Convert { path, custom } => convert(&path, &custom).await,
        Cmd::Listen => listen().await,
    }
}

// ------------------------------------------------------------------ commands

async fn list() -> Result<()> {
    match VoicePe::from_env().await {
        Ok(pe) => {
            println!("On the device (running firmware):");
            for m in wake_word::device_models(&pe).await? {
                let cutoff = m
                    .probability_cutoff
                    .map(|c| format!("cutoff {c:.2}"))
                    .unwrap_or_default();
                println!("  {:24} {cutoff}", m.wake_word);
            }
            pe.close().await;
        }
        Err(e) => println!("On the device: can't connect ({e:#})"),
    }

    let stored = stored_models()?;
    if !stored.is_empty() {
        println!("\nStored for the next build (firmware/models/):");
        for (id, m) in &stored {
            println!("  {id:24} \"{}\"", m.wake_word);
        }
    }

    println!("\nOnline catalogue (esphome/micro-wake-word-models, `install <name>`):");
    for dir in ["", "/experiments"] {
        let names = catalogue_names(dir).await?;
        for name in names {
            let url = format!("{CATALOGUE}{dir}/{name}.json");
            let m = Manifest::parse(&download(&url).await?)?;
            println!(
                "  {name:24} \"{}\" {:?}{}",
                m.wake_word,
                m.trained_languages,
                if dir.is_empty() {
                    ""
                } else {
                    "  (experimental)"
                }
            );
        }
    }
    Ok(())
}

async fn listen() -> Result<()> {
    let pe = VoicePe::from_env().await?;
    let mut det = Detections::start(&pe).await?;
    println!("listening for wake words on {} - Ctrl-C to stop", pe.ip());
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    loop {
        let d = tokio::select! {
            _ = &mut ctrl_c => break,
            d = det.next() => d,
        };
        let Some(d) = d else {
            eprintln!("connection closed: {}", pe.client().close_reason());
            break;
        };
        let now = chrono::Local::now().format("%H:%M:%S");
        if d.vad_rejected {
            println!(
                "{now}  ({}?) model fired, but no voice detected - ignored",
                d.wake_word
            );
        } else {
            println!(
                "{now}  ★ {}  average {:.2}, peak {:.2}",
                d.wake_word,
                d.average.unwrap_or(0.0),
                d.max.unwrap_or(0.0)
            );
        }
    }
    pe.close().await;
    Ok(())
}

/// Write the firmware YAML for every stored model; with `flash`, build and upload it.
async fn build(flash: bool) -> Result<()> {
    let key = std::env::var("HAVOICE_KEY")
        .ok()
        .filter(|k| !k.is_empty())
        .context("HAVOICE_KEY is needed: the new firmware must keep the device's API key")?;
    let pe = VoicePe::from_env().await?;
    let info = pe.info().await?;
    let ip = pe.ip();
    pe.close().await;

    // the stock config the device runs, as it announces it over mDNS
    let package_url = discover::discover(Duration::from_secs(3))
        .await
        .ok()
        .and_then(|found| {
            found
                .into_iter()
                .find(|d| d.name == info.name)?
                .txt
                .get("package_import_url")
                .cloned()
        })
        .unwrap_or_else(|| STOCK_PACKAGE.into());

    let models = stored_models()?;
    let fw = Firmware {
        name: info.name.clone(),
        friendly_name: info.friendly_name.clone(),
        package_url,
        models: models
            .iter()
            .map(|(id, _)| (id.clone(), format!("models/{id}.json")))
            .collect(),
    };
    let dir = firmware_dir();
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(YAML), wake_word::firmware_yaml(&fw))?;
    let secrets = dir.join("secrets.yaml");
    write_secret(&secrets, "havoice_api_key", &key)?;
    // fallback hotspot password: generated once, then kept
    let has_ap = std::fs::read_to_string(&secrets)
        .unwrap_or_default()
        .lines()
        .any(|l| l.starts_with("havoice_ap_password:"));
    if !has_ap {
        let pw: String = havoice::api::generate_key()?
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .take(20)
            .collect();
        write_secret(&secrets, "havoice_ap_password", &pw)?;
    }
    println!(
        "wrote firmware/{YAML}: {} + {} extra model(s){}",
        fw.package_url,
        models.len(),
        if models.is_empty() {
            " (= stock firmware)"
        } else {
            ""
        }
    );

    if !flash {
        println!(
            "\nnot flashed. To build and upload it:\n  cargo run --example wake_word -- build --flash\n\
             or by hand:\n  {} run firmware/{YAML} --device {ip} --no-logs",
            esphome_bin().display()
        );
        return Ok(());
    }

    println!("\nbuilding (first build: 10-20 min) ...");
    // A first build sometimes fails to link after ESP-IDF reconfigures itself midway;
    // the same build run again succeeds. Upload only after a good build.
    let yaml = dir.join(YAML);
    if !esphome(&["compile"], &yaml, &[]).await? {
        println!("\nbuild failed - retrying once ...");
        ensure!(
            esphome(&["compile"], &yaml, &[]).await?,
            "firmware build failed - nothing was uploaded, the device runs its previous firmware"
        );
    }
    println!("\nuploading to {ip} over Wi-Fi ...");
    ensure!(
        esphome(&["upload"], &yaml, &["--device", &ip.to_string()]).await?,
        "upload failed - the device keeps its previous firmware (ESPHome only switches after a complete upload)"
    );

    println!("\nuploaded; waiting for the device to restart ...");
    let pe = reconnect().await?;
    let info = pe.info().await?;
    println!(
        "back online: {} (ESPHome {})",
        info.friendly_name, info.esphome_version
    );
    println!("models now in the firmware:");
    for m in wake_word::device_models(&pe).await? {
        println!("  {}", m.wake_word);
    }
    pe.close().await;
    println!("\nsay a wake word and watch:  cargo run --example wake_word -- listen");
    Ok(())
}

async fn reconnect() -> Result<VoicePe> {
    let mut last = None;
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_secs(3)).await;
        match VoicePe::from_env().await {
            Ok(pe) => return Ok(pe),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap().context("device didn't come back within 90 s"))
}

// ------------------------------------------------------------------ onnx

/// Inspect an ONNX model, run onnx2tf on it, and check every .tflite it produces.
async fn convert(path: &Path, custom: &Custom) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let info = onnx::inspect_onnx(&bytes).context("not a readable ONNX model")?;
    println!(
        "{}: ONNX ir {}, producer {}, opsets {}",
        path.display(),
        info.ir_version,
        if info.producer.is_empty() {
            "?"
        } else {
            &info.producer
        },
        info.opsets
            .iter()
            .map(|(d, v)| format!("{}:{}", if d.is_empty() { "ai.onnx" } else { d }, v))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "  ops: {}",
        info.ops
            .iter()
            .map(|(op, n)| format!("{op}x{n}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    for t in info.inputs.iter().chain(&info.outputs) {
        println!(
            "  {} {} {:?}",
            if info.inputs.contains(t) {
                "in "
            } else {
                "out"
            },
            t.elem_type,
            t.shape
        );
    }
    println!(
        "  {} weight tensor(s), {} kB\n",
        info.initializers,
        info.parameter_bytes / 1000
    );
    let onnx_checks = onnx::checks(&info);
    print_checks(&onnx_checks);
    let hopeless = onnx_checks.iter().any(|c| !c.ok && !c.warning);
    ensure!(
        !hopeless,
        "this model can't become a microWakeWord model as it is - see the \u{2717} above"
    );

    check_onnx2tf().await?;
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let outdir = firmware_dir().join(format!("onnx2tf-{stem}"));
    if outdir.exists() {
        std::fs::remove_dir_all(&outdir)?;
    }
    println!(
        "\nconverting with onnx2tf (int8 quantization) - its warnings are usually harmless ..."
    );
    let status = tokio::process::Command::new("onnx2tf")
        .arg("-i")
        .arg(path)
        .arg("-o")
        .arg(&outdir)
        .arg("-oiqt")
        .status()
        .await
        .context("running onnx2tf")?;
    ensure!(status.success(), "onnx2tf failed - see its output above");

    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&outdir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "tflite"))
        .collect();
    candidates.sort_by_key(|p| {
        let n = p.file_name().unwrap().to_string_lossy().to_lowercase();
        (!n.contains("integer") && !n.contains("quant"), n.clone())
    });
    ensure!(
        !candidates.is_empty(),
        "onnx2tf produced no .tflite in {}",
        outdir.display()
    );

    let esphome = esphome_version().await.ok();
    let mut any_ok = false;
    for c in &candidates {
        let tflite = std::fs::read(c)?;
        let manifest = apply(
            bare_manifest(&c.file_name().unwrap().to_string_lossy(), custom)?,
            custom,
        );
        let m = Model {
            manifest,
            tflite,
            source: c.display().to_string(),
        };
        any_ok |= print_report(&m, &check(&m, esphome.as_deref()));
    }
    if any_ok {
        let best = &candidates[0];
        println!(
            "a candidate passed - install it with:\n  cargo run --example wake_word -- install {} --wake-word \"{}\"",
            best.display(),
            custom
                .wake_word
                .clone()
                .unwrap_or_else(|| "Your Phrase".into())
        );
        println!(
            "\nconversion is lossy: say the phrase to the device afterwards and check\n  cargo run --example wake_word -- listen"
        );
    } else {
        println!(
            "no candidate passed the device's checks. Common reasons:\n\
             - the model wasn't trained as a streaming microWakeWord network (most ONNX models aren't)\n\
             - quantization needs real calibration audio: onnx2tf -cind <input_name>:<npy path>\n\
             - an op can't be lowered to one of the 20 the device's interpreter has"
        );
    }
    Ok(())
}

/// `python3 -c "import onnx2tf, tensorflow"`, with an actionable error.
async fn check_onnx2tf() -> Result<()> {
    match tokio::process::Command::new("python3")
        .args(["-c", "import onnx2tf, tensorflow"])
        .output()
        .await
    {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!("python3 not found - onnx2tf runs through Python")
        }
        Err(e) => bail!(e),
        Ok(out) if out.status.success() => Ok(()),
        Ok(_) => bail!(
            "onnx2tf/tensorflow not importable - install them with:\n  pip install onnx2tf tensorflow"
        ),
    }
}

/// Print checks with the same marks as `print_report`.
fn print_checks(checks: &[Check]) {
    for c in checks {
        let mark = if c.ok {
            "\u{2713}"
        } else if c.warning {
            "!"
        } else {
            "\u{2717}"
        };
        println!("  {mark} {}", c.what);
    }
}

// ------------------------------------------------------------------ models

/// Fetch a model from a catalogue name, URL or local path.
async fn load(source: &str, custom: &Custom) -> Result<Model> {
    let (manifest, tflite) = if Path::new(source).exists() {
        load_local(Path::new(source), custom)?
    } else if source.starts_with("http://") || source.starts_with("https://") {
        load_url(source, custom).await?
    } else {
        // catalogue: stable models first, then experiments
        let mut found = None;
        for dir in ["", "/experiments"] {
            let url = format!("{CATALOGUE}{dir}/{source}.json");
            if let Ok(m) = load_url(&url, custom).await {
                found = Some(m);
                break;
            }
        }
        found.with_context(|| {
            format!("{source:?} is no file, URL or catalogue model (see `wake_word list`)")
        })?
    };
    Ok(Model {
        manifest: apply(manifest, custom),
        tflite,
        source: source.into(),
    })
}

fn load_local(path: &Path, custom: &Custom) -> Result<(Manifest, Vec<u8>)> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "json" {
        let manifest = Manifest::parse(&std::fs::read(path)?)?;
        let tflite_path = path.with_file_name(&manifest.model);
        let tflite = std::fs::read(&tflite_path)
            .with_context(|| format!("manifest points to {}", tflite_path.display()))?;
        return Ok((manifest, tflite));
    }
    let tflite = std::fs::read(path)?;
    let sidecar = path.with_extension("json");
    let manifest = if sidecar.exists() {
        Manifest::parse(&std::fs::read(&sidecar)?)?
    } else {
        bare_manifest(path.file_name().unwrap().to_string_lossy().as_ref(), custom)?
    };
    Ok((manifest, tflite))
}

async fn load_url(url: &str, custom: &Custom) -> Result<(Manifest, Vec<u8>)> {
    let base = &url[..url.rfind('/').map_or(0, |i| i + 1)];
    if url.ends_with(".json") {
        let manifest = Manifest::parse(&download(url).await?)?;
        let tflite = download(&format!("{base}{}", manifest.model)).await?;
        return Ok((manifest, tflite));
    }
    let tflite = download(url).await?;
    let file = url.rsplit('/').next().unwrap_or("model.tflite");
    // a manifest usually sits next to the model
    let sidecar = format!("{}.json", url.trim_end_matches(".tflite"));
    let manifest = match download(&sidecar).await.ok().map(|j| Manifest::parse(&j)) {
        Some(Ok(m)) => m,
        _ => bare_manifest(file, custom)?,
    };
    Ok((manifest, tflite))
}

fn bare_manifest(file: &str, custom: &Custom) -> Result<Manifest> {
    let word = custom.wake_word.as_deref().with_context(|| {
        format!("{file} came without a manifest - give the phrase with --wake-word \"...\"")
    })?;
    Ok(Manifest::for_custom(word, file, custom.language.as_deref()))
}

fn apply(mut m: Manifest, c: &Custom) -> Manifest {
    if let Some(w) = &c.wake_word {
        m.wake_word = w.clone();
    }
    if let Some(l) = &c.language {
        m.trained_languages = vec![l.clone()];
    }
    if let Some(v) = c.cutoff {
        m.micro.probability_cutoff = v;
    }
    if let Some(v) = c.window {
        m.micro.sliding_window_size = v;
    }
    if let Some(v) = c.arena {
        m.micro.tensor_arena_size = v;
    }
    m
}

fn check(m: &Model, esphome: Option<&str>) -> Report {
    wake_word::verify(&m.manifest, &m.tflite, esphome)
}

/// Print the report; true if it passed.
fn print_report(m: &Model, r: &Report) -> bool {
    println!(
        "{}  \"{}\" {:?}",
        m.source, m.manifest.wake_word, m.manifest.trained_languages
    );
    println!("  {} bytes, sha256 {}", r.size, r.sha256);
    if let Some(t) = &r.tflite {
        println!(
            "  input {:?} {}, output {:?} {}, {} subgraph(s), ops: {}",
            t.input_shape,
            t.input_type,
            t.output_shape,
            t.output_type,
            t.subgraphs,
            t.operators.join(" ")
        );
    }
    for c in &r.checks {
        let mark = if c.ok {
            "✓"
        } else if c.warning {
            "!"
        } else {
            "✗"
        };
        println!("  {mark} {}", c.what);
    }
    let ok = r.passed();
    println!("  => {}\n", if ok { "OK" } else { "FAILED" });
    ok
}

/// Store a verified model as `firmware/models/<id>.json` + `.tflite`.
fn store(m: &Model) -> Result<String> {
    let id = wake_word::model_id(&m.manifest.wake_word);
    let dir = models_dir();
    std::fs::create_dir_all(&dir)?;
    let mut manifest = m.manifest.clone();
    manifest.model = format!("{id}.tflite");
    std::fs::write(dir.join(format!("{id}.tflite")), &m.tflite)?;
    std::fs::write(
        dir.join(format!("{id}.json")),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(id)
}

fn stored_models() -> Result<Vec<(String, Manifest)>> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(models_dir()) else {
        return Ok(out);
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "json") {
            let id = p.file_stem().unwrap().to_string_lossy().into_owned();
            out.push((id, Manifest::parse(&std::fs::read(&p)?)?));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

// ------------------------------------------------------------------ helpers

fn firmware_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("firmware")
}

fn models_dir() -> PathBuf {
    firmware_dir().join("models")
}

fn esphome_bin() -> PathBuf {
    if let Ok(p) = std::env::var("HAVOICE_ESPHOME")
        && !p.is_empty()
    {
        return p.into();
    }
    let venv = Path::new(env!("CARGO_MANIFEST_DIR")).join(".venv-esphome/bin/esphome");
    if venv.exists() {
        venv
    } else {
        "esphome".into()
    }
}

/// Run `esphome <cmd> <yaml> <extra>`; true on success. Output goes straight to the terminal.
async fn esphome(cmd: &[&str], yaml: &Path, extra: &[&str]) -> Result<bool> {
    let status = tokio::process::Command::new(esphome_bin())
        .args(cmd)
        .arg(yaml)
        .args(extra)
        .status()
        .await
        .context("running the ESPHome CLI - install it with `pip install esphome`")?;
    Ok(status.success())
}

/// `2026.10.2` from `esphome version`.
async fn esphome_version() -> Result<String> {
    let out = tokio::process::Command::new(esphome_bin())
        .arg("version")
        .output()
        .await?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace()
        .last()
        .map(String::from)
        .context("no version from `esphome version`")
}

/// Set one key in an ESPHome secrets file, keeping the others; owner-only permissions.
fn write_secret(path: &Path, name: &str, value: &str) -> Result<()> {
    let old = std::fs::read_to_string(path).unwrap_or_default();
    let mut lines: Vec<String> = old
        .lines()
        .filter(|l| !l.starts_with(&format!("{name}:")))
        .map(String::from)
        .collect();
    lines.push(format!("{name}: \"{value}\""));
    std::fs::write(path, lines.join("\n") + "\n")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

async fn download(url: &str) -> Result<Vec<u8>> {
    let resp = reqwest::Client::new()
        .get(url)
        .header("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .with_context(|| format!("downloading {url}"))?
        .error_for_status()
        .with_context(|| format!("downloading {url}"))?;
    Ok(resp.bytes().await?.to_vec())
}

/// Model names (`*.json`) in a catalogue folder.
async fn catalogue_names(dir: &str) -> Result<Vec<String>> {
    let listing: serde_json::Value =
        serde_json::from_slice(&download(&format!("{CATALOGUE_DIRS}{dir}")).await?)?;
    let mut names: Vec<String> = listing
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|f| f["name"].as_str()?.strip_suffix(".json").map(String::from))
        .filter(|n| n != "vad")
        .collect();
    names.sort();
    Ok(names)
}
