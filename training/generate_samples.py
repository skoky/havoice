#!/usr/bin/env python3
"""Generate Czech "Ahoj Hugo" training samples for microWakeWord.

The device firmware cannot be extended from Rust and piper-sample-generator
ships no Czech voice checkpoint, so samples are synthesized with the Piper
TTS voices themselves (cs_CZ jirka + kasandra, downloaded on first run),
optionally mixed with Microsoft edge-tts and macOS `say` for extra variety.

    python3 -m pip install piper-tts
    python3 training/generate_samples.py --count 1000 --out samples/ahoj_hugo
    python3 training/generate_samples.py --edge 300 --say 100
    python3 training/generate_samples.py --extra-wavs ../oww_rs/crates/oww/test_data
    python3 training/generate_samples.py --jobs 8 --edge-jobs 4

Piper samples are synthesized in parallel worker processes (--jobs, default:
all CPU cores) and edge-tts/say calls run in parallel threads. Each sample's
voice, phrase, speed and noise settings are drawn up front from --seed, so they
don't depend on the job count. Piper adds its own unseeded noise on top, so
the files differ slightly between runs, as they always have.

Output is 16 kHz mono 16-bit WAVs (what the microWakeWord feature pipeline
expects) plus a ready-to-upload zip for training/train_ahoj_hugo.ipynb.
"""

import argparse
import importlib.util
import os
import platform
import random
import shutil
import subprocess
import sys
import urllib.request
import wave
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

HUGGINGFACE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICES = {
    "jirka": f"{HUGGINGFACE}/cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium.onnx",
    "kasandra": f"{HUGGINGFACE}/cs/cs_CZ/kasandra/medium/cs_CZ-kasandra-medium.onnx",
}
PHRASES = ["Ahoj Hugo.", "Ahoj Hugo!", "Ahoj Hugo?", "Ahoj, Hugo.", "No ahoj, Hugo."]
EDGE_VOICES = ["cs-CZ-AntoninNeural", "cs-CZ-VlastaNeural"]
TARGET_RATE = 16000


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, part)
    part.rename(dest)


def resample_to_16k(src: Path, dst: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
             "-ar", str(TARGET_RATE), "-ac", "1", "-sample_fmt", "s16", str(dst)],
            check=True,
        )
        return
    try:
        import numpy as np
        from scipy.io import wavfile
        from scipy.signal import resample_poly
    except ImportError:
        sys.exit(f"cannot resample {src}: install ffmpeg (brew install ffmpeg) or scipy")
    rate, data = wavfile.read(src)
    if data.ndim > 1:
        data = data.mean(axis=1)
    resampled = resample_poly(data.astype(np.float64), TARGET_RATE, rate)
    wavfile.write(dst, TARGET_RATE, resampled.astype(np.int16))


def ensure_voice(name: str, cache: Path) -> tuple[Path, Path]:
    onnx = cache / f"cs_CZ-{name}-medium.onnx"
    config = onnx.with_suffix(".onnx.json")
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(download, [VOICES[name], VOICES[name] + ".json"], [onnx, config]))
    return onnx, config


def check_piper() -> None:
    try:
        import piper  # noqa: F401
    except ImportError:
        sys.exit("piper-tts is missing: python3 -m pip install piper-tts")


_worker_voices: dict = {}


def init_piper_worker(voice_files: dict[str, tuple[Path, Path]]) -> None:
    """Load every Piper voice once per worker process, single-threaded so N
    workers use N cores instead of fighting over onnxruntime's thread pool."""
    import onnxruntime
    from piper import PiperVoice

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    for name, (onnx, config) in voice_files.items():
        voice = PiperVoice.load(str(onnx), str(config))
        voice.session = onnxruntime.InferenceSession(
            str(onnx), sess_options=options, providers=["CPUExecutionProvider"])
        _worker_voices[name] = voice


def piper_job(job: tuple) -> None:
    from piper import SynthesisConfig

    voice_name, text, length_scale, noise_scale, noise_w_scale, tmp, out = job
    config = SynthesisConfig(
        length_scale=length_scale, noise_scale=noise_scale, noise_w_scale=noise_w_scale)
    with wave.open(str(tmp), "wb") as wav:
        _worker_voices[voice_name].synthesize_wav(text, wav, syn_config=config)
    resample_to_16k(tmp, out)
    tmp.unlink()


def edge_command() -> list[str] | None:
    """Prefer the edge_tts module of this interpreter, so it works from an
    unactivated venv where `edge-tts` is not on PATH."""
    if importlib.util.find_spec("edge_tts"):
        return [sys.executable, "-m", "edge_tts"]
    found = shutil.which("edge-tts")
    return [found] if found else None


def edge_job(job: tuple) -> bool:
    command, text, voice, rate, pitch, tmp_mp3, out = job
    result = subprocess.run(
        [*command, "--voice", voice, "--text", text,
         f"--rate={rate}", f"--pitch={pitch}", "--write-media", str(tmp_mp3)],
        capture_output=True,
    )
    if result.returncode != 0 or not tmp_mp3.exists():
        return False
    resample_to_16k(tmp_mp3, out)
    tmp_mp3.unlink()
    return True


def say_job(job: tuple) -> bool:
    text, rate, tmp, out = job
    result = subprocess.run(
        ["say", "-v", "Zuzana", "-r", str(rate),
         "-o", str(tmp), "--data-format=LEI16@16000", text],
        capture_output=True,
    )
    if result.returncode != 0 or not tmp.exists():
        return False
    resample_to_16k(tmp, out)
    tmp.unlink()
    return True


def fold_in_extra_wavs(directory: Path, out_dir: Path) -> int:
    count = 0
    for src in sorted(directory.glob("*.wav")):
        try:
            resample_to_16k(src, out_dir / f"extra_{src.stem}.wav")
            count += 1
        except Exception as error:
            print(f"skipping {src.name}: {error}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1000, help="Piper samples to generate (default 1000)")
    parser.add_argument("--edge", type=int, default=0,
                        help="extra samples via edge-tts (needs `pip install edge-tts`)")
    parser.add_argument("--say", type=int, default=0, help="extra samples via macOS `say -v Zuzana`")
    parser.add_argument("--extra-wavs", type=Path, default=None,
                        help="fold real recordings in, e.g. ../oww_rs/crates/oww/test_data")
    parser.add_argument("--out", type=Path, default=Path("samples/ahoj_hugo"))
    parser.add_argument("--voice-cache", type=Path, default=Path("samples/voices"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="parallel Piper worker processes (default: all CPU cores)")
    parser.add_argument("--edge-jobs", type=int, default=4,
                        help="parallel edge-tts requests (default 4; more risks throttling)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out.parent / "tmp"
    tmp.mkdir(exist_ok=True)

    # Draw every random parameter up front, in the same order as the old
    # sequential loop, so they depend only on --seed and not on scheduling.
    voice_names = list(VOICES)
    piper_jobs = []
    for i in range(args.count):
        voice_name = rng.choice(voice_names)
        text = rng.choice(PHRASES)
        piper_jobs.append((voice_name, text, rng.uniform(0.65, 1.4), rng.uniform(0.5, 0.8),
                           rng.uniform(0.6, 1.0), tmp / f"piper_{i:05d}.wav",
                           args.out / f"{i:05d}.wav"))
    edge_jobs = [(rng.choice(PHRASES), rng.choice(EDGE_VOICES), f"{rng.randint(-35, 35):+d}%",
                  f"{rng.randint(-15, 15):+d}Hz", tmp / f"edge_{i:05d}.mp3",
                  args.out / f"edge_{i:05d}.wav") for i in range(args.edge)]
    say_jobs = [(rng.choice(PHRASES), rng.randint(160, 210), tmp / f"say_{i:05d}.wav",
                 args.out / f"say_{i:05d}.wav") for i in range(args.say)]

    if piper_jobs:
        check_piper()
        with ThreadPoolExecutor(len(VOICES)) as pool:
            voice_files = dict(zip(VOICES, pool.map(
                lambda name: ensure_voice(name, args.voice_cache), VOICES)))
        jobs = max(1, min(args.jobs, len(piper_jobs)))
        print(f"synthesizing {len(piper_jobs)} Piper samples with {jobs} worker(s)")
        with ProcessPoolExecutor(jobs, initializer=init_piper_worker,
                                 initargs=(voice_files,)) as pool:
            for done, _ in enumerate(pool.map(piper_job, piper_jobs, chunksize=4), 1):
                if done % 100 == 0 or done == len(piper_jobs):
                    print(f"{done}/{len(piper_jobs)} Piper samples")

    if edge_jobs:
        command = edge_command()
        if not command:
            print("edge-tts not usable - install it with `python3 -m pip install edge-tts`")
        else:
            with ThreadPoolExecutor(max(1, args.edge_jobs)) as pool:
                ok = sum(pool.map(edge_job, [(command, *job) for job in edge_jobs]))
            print(f"{ok}/{len(edge_jobs)} edge-tts samples")
            if ok < len(edge_jobs):
                print("some edge-tts requests failed (network or throttling); "
                      "try a lower --edge-jobs")

    if say_jobs:
        if platform.system() != "Darwin":
            print("say is macOS only - skipping --say")
        else:
            with ThreadPoolExecutor(os.cpu_count() or 1) as pool:
                ok = sum(pool.map(say_job, say_jobs))
            print(f"{ok}/{len(say_jobs)} say samples")
            if ok == 0:
                print("say -v Zuzana not usable - add the Czech voice in System Settings")

    if args.extra_wavs:
        folded = fold_in_extra_wavs(args.extra_wavs, args.out)
        print(f"folded in {folded} real recording(s) from {args.extra_wavs}")

    shutil.rmtree(tmp, ignore_errors=True)
    total = len(list(args.out.glob("*.wav")))
    print(f"{total} samples in {args.out}")
    zip_path = shutil.make_archive(str(args.out), "zip", root_dir=args.out, base_dir=".")
    print(f"upload {zip_path} into train_ahoj_hugo.ipynb")


if __name__ == "__main__":
    main()
