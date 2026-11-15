//! Score every WAV in a folder with a wake word model and report the success rate.
//!
//!     cargo run --release --example test_wavs -- samples/extra
//!     cargo run --release --example test_wavs -- samples/extra --tflite my.tflite --cutoff 0.9
//!     cargo run --release --example test_wavs -- samples/extra --onnx ahoj_hugo_model/ahoj_hugo.onnx
//!     cargo run --release --example test_wavs -- samples/negatives --expect silent
//!
//! `.tflite` (microWakeWord, the Voice PE): the device's frontend and int8 kernels, ported
//! to Rust and checked to give the same numbers, then the firmware's rule - the average of
//! the last `--window` probabilities must exceed `--cutoff`; both default to the model's
//! manifest (`<model>.json`). Nothing fires in the first second after a reset, so each
//! clip gets `--pad-start` seconds of silence in front.
//!
//! `.onnx` (openWakeWord classifier, the Hugo assistant): the assistant's feature chain
//! (melspectrogram + embedding models, 80 ms chunks) run with tract, then its rule - more
//! than `--onnx-min-positive` of the last 12 probabilities above `--onnx-threshold`, and
//! their average above it too. The two feature models are downloaded once into
//! `--feature-models`.
//!
//! Both models are tested when both files exist; `--tflite`/`--onnx` pick other files and
//! `--no-tflite`/`--no-onnx` skip one. WAVs may be any sample rate, 8/16/24/32-bit or float,
//! mono or stereo. Files are tested in parallel (`--jobs`, default one per CPU core); the
//! failed files and the success rates are listed at the end. Exits with status 1 when a
//! model scores below `--min-rate`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use havoice::micro_model::{Detector, StreamingModel};
use havoice::wake_word::Manifest;
use sha2::{Digest, Sha256};
use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::IntoRunnable;

const OWW_RELEASE: &str = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/";
// openWakeWord v0.5.1, the same files the assistant embeds (training/train_oww.py)
const FEATURE_MODELS: [(&str, &str); 2] = [
    (
        "melspectrogram.onnx",
        "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    ),
    (
        "embedding_model.onnx",
        "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    ),
];
const CHUNK: usize = 1280; // 80 ms, the assistant's step
const DETECTION_BUFFER: usize = 12;
const REFRACTORY_CHUNKS: usize = 2000 / 80;

#[derive(Parser)]
#[command(about = "Score every WAV in a folder with a wake word model and report the success rate")]
struct Cli {
    /// folder with the WAV files to test
    dir: PathBuf,
    /// microWakeWord model; its manifest is read from the .json next to it
    #[arg(long, default_value = "ahoj_hugo_model/ahoj_hugo.tflite")]
    tflite: PathBuf,
    /// openWakeWord classifier for the Hugo assistant
    #[arg(long, default_value = "ahoj_hugo_model/ahoj_hugo.onnx")]
    onnx: PathBuf,
    /// skip the .tflite model
    #[arg(long)]
    no_tflite: bool,
    /// skip the .onnx model
    #[arg(long)]
    no_onnx: bool,
    /// .tflite probability cutoff 0-1 (default: the manifest's probability_cutoff)
    #[arg(long)]
    cutoff: Option<f32>,
    /// .tflite sliding window size (default: the manifest's sliding_window_size)
    #[arg(long)]
    window: Option<usize>,
    /// .onnx probability threshold 0-1
    #[arg(long, default_value_t = 0.5)]
    onnx_threshold: f32,
    /// .onnx: more than this many of the last 12 probabilities must pass the threshold
    #[arg(long, default_value_t = 2)]
    onnx_min_positive: usize,
    /// what every file should do: wake the model, or stay silent (for negative recordings)
    #[arg(long, value_enum, default_value_t = Expect::Detect)]
    expect: Expect,
    /// seconds of silence before each clip (the device ignores its first second)
    #[arg(long, default_value_t = 1.0)]
    pad_start: f32,
    /// seconds of silence after each clip, so a phrase at the very end can still fire
    #[arg(long, default_value_t = 0.5)]
    pad_end: f32,
    /// folder with (or for) openWakeWord's melspectrogram.onnx and embedding_model.onnx
    #[arg(long, default_value = "training_run/oww/models")]
    feature_models: PathBuf,
    /// exit with status 1 if a model's success rate (percent) is below this
    #[arg(long, default_value_t = 100.0)]
    min_rate: f32,
    /// files tested at the same time (default: one per CPU core)
    #[arg(long)]
    jobs: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Expect {
    Detect,
    Silent,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let files = wav_files(&cli.dir)?;
    ensure!(!files.is_empty(), "no .wav files in {}", cli.dir.display());

    let mut tflite: Option<(Detector, f32, usize)> = None;
    if !cli.no_tflite {
        let bytes = std::fs::read(&cli.tflite).with_context(|| {
            format!("reading {} (--no-tflite to skip it)", cli.tflite.display())
        })?;
        let manifest = std::fs::read(cli.tflite.with_extension("json"))
            .ok()
            .and_then(|j| serde_json::from_slice::<Manifest>(&j).ok());
        let cutoff = cli
            .cutoff
            .or(manifest.as_ref().map(|m| m.micro.probability_cutoff))
            .unwrap_or(0.97);
        let window = cli
            .window
            .or(manifest
                .as_ref()
                .map(|m| m.micro.sliding_window_size as usize))
            .unwrap_or(5);
        let model = StreamingModel::load(&bytes)
            .with_context(|| format!("loading {}", cli.tflite.display()))?;
        println!(
            "tflite {}  cutoff {cutoff:.2}, window {window}",
            cli.tflite.display()
        );
        tflite = Some((Detector::new(model, cutoff, window), cutoff, window));
    }
    let mut onnx: Option<Oww> = None;
    if !cli.no_onnx {
        ensure!(
            cli.onnx.is_file(),
            "{} not found (--no-onnx to skip it)",
            cli.onnx.display()
        );
        feature_models(&cli.feature_models).await?;
        println!(
            "onnx   {}  threshold {:.2}, more than {} of the last {DETECTION_BUFFER} above it",
            cli.onnx.display(),
            cli.onnx_threshold,
            cli.onnx_min_positive
        );
        onnx = Some(Oww::load(&cli.feature_models, &cli.onnx)?);
    }
    ensure!(
        tflite.is_some() || onnx.is_some(),
        "nothing to test: both models skipped"
    );

    let want = cli.expect == Expect::Detect;
    let jobs = cli
        .jobs
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .clamp(1, files.len());
    println!(
        "\n{} WAVs in {}, each should {}, {jobs} in parallel\n",
        files.len(),
        cli.dir.display(),
        if want {
            "wake the model"
        } else {
            "stay silent"
        }
    );

    // every worker gets its own copy of the models and takes the next file
    let next = AtomicUsize::new(0);
    let results: Vec<Outcome> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..jobs)
            .map(|_| {
                let mut detector = tflite.as_ref().map(|(d, _, _)| d.clone());
                let mut oww = onnx.clone();
                let (cli, files, next) = (&cli, &files, &next);
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(file) = files.get(i) else {
                            return done;
                        };
                        let outcome = test_file(file, cli, want, detector.as_mut(), oww.as_mut());
                        println!("{}", outcome.line);
                        done.push((i, outcome));
                    }
                })
            })
            .collect();
        let mut all: Vec<(usize, Outcome)> = workers
            .into_iter()
            .flat_map(|w| w.join().expect("worker thread panicked"))
            .collect();
        all.sort_by_key(|(i, _)| *i);
        all.into_iter().map(|(_, o)| o).collect()
    });

    let total = files.len();
    let tflite_ok = results
        .iter()
        .filter(|r| r.tflite_pass == Some(true))
        .count();
    let onnx_ok = results.iter().filter(|r| r.onnx_pass == Some(true)).count();
    let failed: Vec<&str> = results.iter().filter_map(|r| r.failed.as_deref()).collect();
    if !failed.is_empty() {
        println!("\nfailed ({}):", failed.len());
        for f in &failed {
            println!("  {f}");
        }
    }

    let rate = |ok: usize| 100.0 * ok as f32 / total as f32;
    let verb = if want { "detected" } else { "stayed silent on" };
    println!();
    let mut below = false;
    if let Some((_, cutoff, window)) = &tflite {
        println!(
            "tflite {verb} {tflite_ok}/{total} = {:.1}%   (cutoff {cutoff:.2}, window {window})",
            rate(tflite_ok)
        );
        below |= rate(tflite_ok) < cli.min_rate;
    }
    if onnx.is_some() {
        println!(
            "onnx   {verb} {onnx_ok}/{total} = {:.1}%   (threshold {:.2}, min positive {})",
            rate(onnx_ok),
            cli.onnx_threshold,
            cli.onnx_min_positive
        );
        below |= rate(onnx_ok) < cli.min_rate;
    }
    if below {
        std::process::exit(1);
    }
    Ok(())
}

/// One file's result: its report line, and whether each model did what was expected.
struct Outcome {
    line: String,
    tflite_pass: Option<bool>,
    onnx_pass: Option<bool>,
    /// `name (models that failed)`, if any did
    failed: Option<String>,
}

fn test_file(
    file: &Path,
    cli: &Cli,
    want: bool,
    detector: Option<&mut Detector>,
    oww: Option<&mut Oww>,
) -> Outcome {
    let name = file.file_name().unwrap().to_string_lossy();
    let mut outcome = Outcome {
        line: format!("{name:48}"),
        tflite_pass: None,
        onnx_pass: None,
        failed: None,
    };
    let clip = match read_wav_16k(file) {
        Ok(c) => c,
        Err(e) => {
            outcome.line += &format!("unreadable: {e:#}");
            outcome.failed = Some(format!("{name} (unreadable)"));
            return outcome;
        }
    };
    let pad = |s: f32| vec![0i16; (s.max(0.0) * 16_000.0) as usize];
    let mut audio = pad(cli.pad_start);
    audio.extend(&clip);
    audio.extend(pad(cli.pad_end));
    let mut bad = Vec::new();

    if let Some(detector) = detector {
        match detector.run(&audio) {
            Ok(r) => {
                let pass = r.detected() == want;
                outcome.tflite_pass = Some(pass);
                if !pass {
                    bad.push("tflite");
                }
                let when = r
                    .detections
                    .first()
                    .map(|t| format!(" @{:.2}s", t - cli.pad_start))
                    .unwrap_or_default();
                outcome.line += &format!(
                    "  tflite {} avg {:.2} peak {:.2}{when:8}",
                    mark(r.detected(), pass),
                    r.max_average,
                    r.max_probability
                );
            }
            Err(e) => {
                outcome.tflite_pass = Some(false);
                bad.push("tflite error");
                outcome.line += &format!("  tflite error: {e:#}");
            }
        }
    }
    if let Some(oww) = oww {
        match oww.probabilities(&audio) {
            Ok(probs) => {
                let (fired, peak_avg) =
                    oww_detect(&probs, cli.onnx_threshold, cli.onnx_min_positive);
                let pass = fired.is_some() == want;
                outcome.onnx_pass = Some(pass);
                if !pass {
                    bad.push("onnx");
                }
                let peak = probs.iter().copied().fold(0.0f32, f32::max);
                let when = fired
                    .map(|i| {
                        format!(
                            " @{:.2}s",
                            ((i + 1) * CHUNK) as f32 / 16_000.0 - cli.pad_start
                        )
                    })
                    .unwrap_or_default();
                outcome.line += &format!(
                    "  onnx {} avg {:.2} peak {:.2}{when}",
                    mark(fired.is_some(), pass),
                    peak_avg,
                    peak
                );
            }
            Err(e) => {
                outcome.onnx_pass = Some(false);
                bad.push("onnx error");
                outcome.line += &format!("  onnx error: {e:#}");
            }
        }
    }
    if !bad.is_empty() {
        outcome.failed = Some(format!("{name} ({})", bad.join(", ")));
    }
    outcome
}

fn mark(detected: bool, pass: bool) -> &'static str {
    match (detected, pass) {
        (true, true) => "DETECTED",
        (false, true) => "silent  ",
        (true, false) => "FALSE!  ",
        (false, false) => "MISSED  ",
    }
}

fn wav_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")))
        .collect();
    files.sort();
    Ok(files)
}

// ------------------------------------------------------------------------------ WAV input

/// Any common WAV as 16 kHz mono 16-bit samples.
fn read_wav_16k(path: &Path) -> Result<Vec<i16>> {
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a WAV file"
    );
    let (mut format, mut channels, mut rate, mut bits) = (0u16, 0usize, 0u32, 0u16);
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        match id {
            b"fmt " => {
                ensure!(body.len() >= 16, "short fmt chunk");
                format = u16::from_le_bytes([body[0], body[1]]);
                channels = u16::from_le_bytes([body[2], body[3]]) as usize;
                rate = u32::from_le_bytes(body[4..8].try_into()?);
                bits = u16::from_le_bytes([body[14], body[15]]);
                if format == 0xFFFE && body.len() >= 26 {
                    format = u16::from_le_bytes([body[24], body[25]]); // WAVE_FORMAT_EXTENSIBLE
                }
            }
            b"data" => {
                ensure!(channels > 0 && rate > 0, "data before fmt chunk");
                let width = (bits as usize).div_ceil(8);
                let frame = width * channels;
                let sample = |b: &[u8]| -> Result<f32> {
                    Ok(match (format, bits) {
                        (1, 8) => (b[0] as f32 - 128.0) / 128.0,
                        (1, 16) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
                        (1, 24) => {
                            (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0
                        }
                        (1, 32) => i32::from_le_bytes(b[..4].try_into()?) as f32 / 2_147_483_648.0,
                        (3, 32) => f32::from_le_bytes(b[..4].try_into()?),
                        _ => bail!("unsupported WAV format {format}, {bits}-bit"),
                    })
                };
                let mut mono = Vec::with_capacity(body.len() / frame);
                for f in body.chunks_exact(frame) {
                    let mut sum = 0.0;
                    for c in 0..channels {
                        sum += sample(&f[c * width..(c + 1) * width])?;
                    }
                    mono.push(sum / channels as f32);
                }
                let mono = if rate == 16_000 {
                    mono
                } else {
                    resample(&mono, rate, 16_000)
                };
                return Ok(mono
                    .iter()
                    .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
                    .collect());
            }
            _ => {}
        }
        pos += 8 + len + (len & 1);
    }
    bail!("WAV has no data chunk")
}

/// Windowed-sinc resampling (Blackman window, cutoff just below the lower Nyquist).
fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    const HALF_TAPS: i64 = 32;
    let ratio = to as f64 / from as f64;
    let cutoff = 0.95 * ratio.min(1.0);
    let n = (input.len() as f64 * ratio).round() as usize;
    (0..n)
        .map(|i| {
            let center = i as f64 / ratio;
            let first = center.floor() as i64 - HALF_TAPS + 1;
            let mut acc = 0.0;
            let mut norm = 0.0;
            for j in first..first + 2 * HALF_TAPS {
                let x = center - j as f64;
                let w = 0.42
                    + 0.5 * (std::f64::consts::PI * x / HALF_TAPS as f64).cos()
                    + 0.08 * (2.0 * std::f64::consts::PI * x / HALF_TAPS as f64).cos();
                let arg = std::f64::consts::PI * cutoff * x;
                let sinc = if arg.abs() < 1e-9 {
                    1.0
                } else {
                    arg.sin() / arg
                };
                let k = cutoff * sinc * w.max(0.0);
                norm += k;
                if j >= 0 && (j as usize) < input.len() {
                    acc += k * input[j as usize] as f64;
                }
            }
            (acc / norm) as f32
        })
        .collect()
}

// -------------------------------------------------------------- openWakeWord (assistant)

type Plan = Arc<TypedRunnableModel>;

/// The assistant's openWakeWord chain (backend/unlock/src/oww/audio.rs, oww_model.rs).
#[derive(Clone)]
struct Oww {
    mel: Plan,
    emb: Plan,
    classifier: Plan,
}

fn plan(path: &Path, shape: &[usize]) -> Result<Plan> {
    tract_onnx::onnx()
        .model_for_path(path)
        .with_context(|| format!("loading {}", path.display()))?
        .with_input_fact(0, f32::fact(shape).into())?
        .into_optimized()?
        .into_runnable()
}

impl Oww {
    fn load(features: &Path, classifier: &Path) -> Result<Self> {
        Ok(Self {
            mel: plan(&features.join("melspectrogram.onnx"), &[1, CHUNK])?,
            emb: plan(&features.join("embedding_model.onnx"), &[1, 76, 32, 1])?,
            classifier: plan(classifier, &[1, 16, 96])?,
        })
    }

    /// One probability per 80 ms chunk, from fresh (zero-filled) buffers.
    fn probabilities(&mut self, audio: &[i16]) -> Result<Vec<f32>> {
        let mut mels: VecDeque<Vec<f32>> = (0..16).map(|_| vec![0.0; 5 * 32]).collect();
        let mut embeddings: VecDeque<Vec<f32>> = (0..16).map(|_| vec![0.0; 96]).collect();
        let mut probs = Vec::new();
        for chunk in audio.as_chunks::<CHUNK>().0 {
            let samples: Vec<f32> = chunk.iter().map(|&s| s as f32 / 32768.0).collect();
            let out = self
                .mel
                .run(tvec!(Tensor::from_shape(&[1, CHUNK], &samples)?.into()))?;
            let mel: Vec<f32> = floats(&out[0])?.iter().map(|v| v / 10.0 + 2.0).collect();
            ensure!(
                mel.len() == 5 * 32,
                "melspectrogram gave {} values",
                mel.len()
            );
            mels.pop_front();
            mels.push_back(mel);
            let frames: Vec<f32> = mels.iter().flatten().copied().skip(4 * 32).collect();
            let out = self
                .emb
                .run(tvec!(Tensor::from_shape(&[1, 76, 32, 1], &frames)?.into()))?;
            embeddings.pop_front();
            embeddings.push_back(floats(&out[0])?);
            let window: Vec<f32> = embeddings.iter().flatten().copied().collect();
            let out = self
                .classifier
                .run(tvec!(Tensor::from_shape(&[1, 16, 96], &window)?.into()))?;
            probs.push(
                *floats(&out[0])?
                    .first()
                    .context("classifier gave no output")?,
            );
        }
        Ok(probs)
    }
}

/// A model output as plain floats.
fn floats(t: &TValue) -> Result<Vec<f32>> {
    Ok(t.cast_to::<f32>()?
        .to_plain_array_view::<f32>()?
        .iter()
        .copied()
        .collect())
}

/// The assistant's rule: the first chunk it fires on, and the highest average it reached.
fn oww_detect(probs: &[f32], threshold: f32, min_positive: usize) -> (Option<usize>, f32) {
    let mut buffer: VecDeque<f32> = VecDeque::with_capacity(DETECTION_BUFFER);
    let (mut first, mut best) = (None, 0.0f32);
    let mut last: Option<usize> = None;
    for (i, &p) in probs.iter().enumerate() {
        if buffer.len() == DETECTION_BUFFER {
            buffer.pop_front();
        }
        buffer.push_back(p);
        let hits: Vec<f32> = buffer.iter().copied().filter(|&v| v > threshold).collect();
        let average = if hits.len() > min_positive {
            hits.iter().sum::<f32>() / hits.len() as f32
        } else {
            0.0
        };
        best = best.max(average);
        if average > threshold && last.is_none_or(|l| i - l > REFRACTORY_CHUNKS) {
            first.get_or_insert(i);
            last = Some(i);
            buffer.clear();
        }
    }
    (first, best)
}

/// Make sure the two openWakeWord feature models are in `dir`, downloading them if needed.
async fn feature_models(dir: &Path) -> Result<()> {
    for (name, digest) in FEATURE_MODELS {
        let path = dir.join(name);
        let ok = |b: &[u8]| hex(&Sha256::digest(b)) == digest;
        if std::fs::read(&path).is_ok_and(|b| ok(&b)) {
            continue;
        }
        println!("downloading {name} into {}", dir.display());
        let bytes = reqwest::get(format!("{OWW_RELEASE}{name}"))
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        ensure!(ok(&bytes), "{name} downloaded with an unexpected SHA-256");
        std::fs::create_dir_all(dir)?;
        std::fs::write(&path, &bytes)?;
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
