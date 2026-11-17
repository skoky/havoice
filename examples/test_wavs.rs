//! Score every WAV in a folder with a microWakeWord model and report the success rate.
//!
//!     cargo run --release --example test_wavs -- recordings --tflite hey_computer_model/hey_computer.tflite
//!     cargo run --release --example test_wavs -- recordings --tflite my.tflite --cutoff 0.9
//!     cargo run --release --example test_wavs -- negatives --tflite my.tflite --expect silent
//!
//! The device's frontend and int8 kernels, ported to Rust and checked to give the same
//! numbers, then the firmware's rule - the average of the last `--window` probabilities must
//! exceed `--cutoff`; both default to the model's manifest (`<model>.json`). Nothing fires in
//! the first second after a reset, so each clip gets `--pad-start` seconds of silence in front.
//!
//! WAVs may be any sample rate, 8/16/24/32-bit or float, mono or stereo. Files are tested in
//! parallel (`--jobs`, default one per CPU core); the failed files and the success rate are
//! listed at the end. Exits with status 1 when the model scores below `--min-rate`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use havoice::micro_model::{Detector, StreamingModel};
use havoice::wake_word::Manifest;

#[derive(Parser)]
#[command(
    about = "Score every WAV in a folder with a microWakeWord model and report the success rate"
)]
struct Cli {
    /// folder with the WAV files to test
    dir: PathBuf,
    /// microWakeWord model; its manifest is read from the .json next to it
    #[arg(long)]
    tflite: PathBuf,
    /// .tflite probability cutoff 0-1 (default: the manifest's probability_cutoff)
    #[arg(long)]
    cutoff: Option<f32>,
    /// .tflite sliding window size (default: the manifest's sliding_window_size)
    #[arg(long)]
    window: Option<usize>,
    /// what every file should do: wake the model, or stay silent (for negative recordings)
    #[arg(long, value_enum, default_value_t = Expect::Detect)]
    expect: Expect,
    /// seconds of silence before each clip (the device ignores its first second)
    #[arg(long, default_value_t = 1.0)]
    pad_start: f32,
    /// seconds of silence after each clip, so a phrase at the very end can still fire
    #[arg(long, default_value_t = 0.5)]
    pad_end: f32,
    /// exit with status 1 if the success rate (percent) is below this
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    let files = wav_files(&cli.dir)?;
    ensure!(!files.is_empty(), "no .wav files in {}", cli.dir.display());

    let bytes =
        std::fs::read(&cli.tflite).with_context(|| format!("reading {}", cli.tflite.display()))?;
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
    let detector = Detector::new(model, cutoff, window);

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

    // every worker gets its own copy of the model and takes the next file
    let next = AtomicUsize::new(0);
    let results: Vec<Outcome> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..jobs)
            .map(|_| {
                let mut detector = detector.clone();
                let (cli, files, next) = (&cli, &files, &next);
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(file) = files.get(i) else {
                            return done;
                        };
                        let outcome = test_file(file, cli, want, &mut detector);
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
    let ok = results.iter().filter(|r| r.pass).count();
    let failed: Vec<&str> = results.iter().filter_map(|r| r.failed.as_deref()).collect();
    if !failed.is_empty() {
        println!("\nfailed ({}):", failed.len());
        for f in &failed {
            println!("  {f}");
        }
    }

    let rate = 100.0 * ok as f32 / total as f32;
    let verb = if want { "detected" } else { "stayed silent on" };
    println!("\ntflite {verb} {ok}/{total} = {rate:.1}%   (cutoff {cutoff:.2}, window {window})");
    if rate < cli.min_rate {
        std::process::exit(1);
    }
    Ok(())
}

/// One file's result: its report line, and whether the model did what was expected.
struct Outcome {
    line: String,
    pass: bool,
    /// `name (why)`, if it failed
    failed: Option<String>,
}

fn test_file(file: &Path, cli: &Cli, want: bool, detector: &mut Detector) -> Outcome {
    let name = file.file_name().unwrap().to_string_lossy();
    let mut outcome = Outcome {
        line: format!("{name:48}"),
        pass: false,
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
    match detector.run(&audio) {
        Ok(r) => {
            outcome.pass = r.detected() == want;
            if !outcome.pass {
                outcome.failed = Some(format!("{name} (tflite)"));
            }
            let when = r
                .detections
                .first()
                .map(|t| format!(" @{:.2}s", t - cli.pad_start))
                .unwrap_or_default();
            outcome.line += &format!(
                "  tflite {} avg {:.2} peak {:.2}{when:8}",
                mark(r.detected(), outcome.pass),
                r.max_average,
                r.max_probability
            );
        }
        Err(e) => {
            outcome.failed = Some(format!("{name} (tflite error)"));
            outcome.line += &format!("  tflite error: {e:#}");
        }
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
