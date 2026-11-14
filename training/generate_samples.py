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

Output is 16 kHz mono 16-bit WAVs (what the microWakeWord feature pipeline
expects) plus a ready-to-upload zip for training/train_ahoj_hugo.ipynb.
"""

import argparse
import platform
import random
import shutil
import subprocess
import sys
import urllib.request
import wave
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
    urllib.request.urlretrieve(url, dest)


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
    download(VOICES[name], onnx)
    download(VOICES[name] + ".json", config)
    return onnx, config


def piper_voices(cache: Path):
    try:
        from piper import PiperVoice
    except ImportError:
        sys.exit("piper-tts is missing: python3 -m pip install piper-tts")
    loaded = []
    for name in VOICES:
        onnx, config = ensure_voice(name, cache)
        loaded.append((PiperVoice.load(str(onnx), str(config)), name))
    return loaded


def synth_piper(voice, text: str, tmp: Path, rng: random.Random) -> None:
    from piper import SynthesisConfig

    config = SynthesisConfig(
        length_scale=rng.uniform(0.65, 1.4),
        noise_scale=rng.uniform(0.5, 0.8),
        noise_w_scale=rng.uniform(0.6, 1.0),
    )
    with wave.open(str(tmp), "wb") as wav:
        voice.synthesize_wav(text, wav, syn_config=config)

def synth_edge(text: str, tmp_mp3: Path, rng: random.Random) -> bool:
    if not shutil.which("edge-tts") and not shutil.which("edge_tts"):
        return False
    voice = rng.choice(EDGE_VOICES)
    rate = f"{rng.randint(-35, 35):+d}%"
    pitch = f"{rng.randint(-15, 15):+d}Hz"
    result = subprocess.run(
        ["edge-tts", "--voice", voice, "--text", text,
         f"--rate={rate}"   , f"--pitch={pitch}", "--write-media", str(tmp_mp3)],
        capture_output=True,
    )
    return result.returncode == 0 and tmp_mp3.exists()


def synth_say(text: str, out: Path, rng: random.Random) -> bool:
    if platform.system() != "Darwin":
        return False
    result = subprocess.run(
        ["say", "-v", "Zuzana", "-r", str(rng.randint(160, 210)),
         "-o", str(out), "--data-format=LEI16@16000", text],
        capture_output=True,
    )
    return result.returncode == 0 and out.exists()


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
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out.parent / "tmp"
    tmp.mkdir(exist_ok=True)

    voices = piper_voices(args.voice_cache)
    for i in range(args.count):
        voice, _ = rng.choice(voices)
        text = rng.choice(PHRASES)
        raw = tmp / "raw.wav"
        synth_piper(voice, text, raw, rng)
        resample_to_16k(raw, args.out / f"{i:05d}.wav")
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{args.count} Piper samples")

    for i in range(args.edge):
        text = rng.choice(PHRASES)
        mp3 = tmp / "edge.mp3"
        if not synth_edge(text, mp3, rng):
            print("edge-tts not usable - install it with `python3 -m pip install edge-tts`")
            break
        resample_to_16k(mp3, args.out / f"edge_{i:05d}.wav")

    for i in range(args.say):
        text = rng.choice(PHRASES)
        wav = tmp / "say.wav"
        if not synth_say(text, wav, rng):
            print("say -v Zuzana not usable - add the Czech voice in System Settings")
            break
        resample_to_16k(wav, args.out / f"say_{i:05d}.wav")

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
