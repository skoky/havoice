#!/usr/bin/env python3
"""Generate wake word training samples for microWakeWord with text-to-speech.

Samples are synthesized with Piper TTS voices (downloaded on first run),
optionally mixed with Microsoft edge-tts and macOS `say` for extra variety:

  --count   single-speaker Piper voices of the --language preset
  --multi   multi-speaker Piper models (hundreds of speakers), a random
            speaker per sample
  --edge    Microsoft edge-tts neural voices (needs internet)
  --say     macOS `say`

--language picks the voices: a preset (see LANGUAGES below: cs, de, en) or
any language with the voices given explicitly (--piper-voice, --multi-voice,
--edge-voice, --say-voice, --espeak). Every Piper voice reads the phrase with
the --espeak phonemes, so voices of a neighbouring language add new timbres
with the target pronunciation (the cs preset uses Slovak and Polish voices).

A share of the synthesized samples (--pitch-prob) is shifted up or down by up
to --pitch-range semitones, pitch and formants together, keeping the
duration: it sounds like a different, deeper or higher voice.

Positive phrases come from --phrase (repeatable). A phrase without final
punctuation is also spoken as a statement, exclamation and question. When
TTS mispronounces a word, spell it the way it is said (e.g. "Gůgl" for
"Google" in Czech) and give edge-tts, which knows more words, its own
--edge-phrase list.

Optional negatives (phrases the model must NOT wake on) use the same voices:
--negatives N --negative-phrase "..." [--negative-phrase "..."] writes them to
--negatives-out, which train_wake_word.py takes with --negatives.

    python3 -m pip install piper-tts
    python3 training/generate_samples.py --phrase "Hey Computer" --count 1000 --multi 2000 --edge 300
    python3 training/generate_samples.py --language de --phrase "Hallo Rechner" --count 1000
    python3 training/generate_samples.py --phrase "Hey Computer" \
        --negatives 1000 --negative-phrase "Hey Commuter"
    python3 training/generate_samples.py --phrase "Hey Computer" --extra-wavs my_recordings

Piper samples are synthesized in parallel worker processes (--jobs, default:
all CPU cores) and edge-tts/say calls run in parallel threads. Each sample's
voice, speaker, phrase, speed, noise and pitch settings are drawn up front from
--seed, so they don't depend on the job count. Piper adds its own unseeded
noise on top, so the files differ slightly between runs.

Output is 16 kHz mono 16-bit WAVs (what the microWakeWord feature pipeline
expects) in samples/<name> (--out, named after the first phrase) plus a zip of
each output folder.
"""

import argparse
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import urllib.request
import wave
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

HUGGINGFACE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
# Voice presets per language. "piper": single-speaker voices and how often
# each is picked; "multi": multi-speaker models (one is picked in proportion
# to its speaker count, then a random speaker in it); "espeak": the phonemes
# every Piper voice reads with. Voices of another language get these
# phonemes, so they say the phrase with the target pronunciation in their own
# timbre. de_DE-mls-medium was tried for Czech and dropped: with Czech
# phonemes it babbles, 3-10 s of speech for a two-word phrase.
LANGUAGES = {
    "cs": {
        "piper": {
            "cs_CZ-jirka-medium": 3,
            "cs_CZ-kasandra-medium": 3,
            "cs_CZ-jirka-low": 1,
            "sk_SK-lili-medium": 1,
            "pl_PL-darkman-medium": 1,
            "pl_PL-gosia-medium": 1,
            "pl_PL-mc_speech-medium": 1,
            "pl_PL-bass-high": 1,
        },
        "multi": ["en_US-libritts_r-medium", "en_GB-vctk-medium"],
        "edge": ["cs-CZ-AntoninNeural", "cs-CZ-VlastaNeural"],
        "say": "Zuzana",
        "espeak": "cs",
    },
    "de": {
        "piper": {
            "de_DE-thorsten-medium": 2,
            "de_DE-kerstin-medium": 2,
            "de_DE-ramona-low": 1,
            "de_DE-karlsson-low": 1,
            "de_DE-eva_k-x_low": 1,
            "de_DE-pavoque-low": 1,
        },
        "multi": ["de_DE-mls-medium", "de_DE-thorsten_emotional-medium"],
        "edge": ["de-DE-ConradNeural", "de-DE-KatjaNeural"],
        "say": "Anna",
        "espeak": "de",
    },
    "en": {
        "piper": {
            "en_US-lessac-medium": 1,
            "en_US-amy-medium": 1,
            "en_US-ryan-medium": 1,
            "en_US-joe-medium": 1,
            "en_US-kristin-medium": 1,
            "en_GB-alan-medium": 1,
            "en_GB-jenny_dioco-medium": 1,
            "en_GB-northern_english_male-medium": 1,
            "en_GB-cori-medium": 1,
        },
        "multi": ["en_US-libritts_r-medium", "en_GB-vctk-medium", "en_US-l2arctic-medium"],
        "edge": ["en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural",
                 "en-GB-RyanNeural", "en-GB-SoniaNeural"],
        "say": "Samantha",
        "espeak": "en-us",
    },
}
TARGET_RATE = 16000


def voice_url(name: str) -> str:
    """cs_CZ-jirka-medium -> .../cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium.onnx"""
    locale, speaker, quality = name.split("-")
    return f"{HUGGINGFACE}/{locale.split('_')[0]}/{locale}/{speaker}/{quality}/{name}.onnx"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, part)
    part.rename(dest)


def ensure_voices(names: list[str], cache: Path) -> dict[str, tuple[Path, Path]]:
    """Download every voice model and config in parallel."""
    files = {name: (cache / f"{name}.onnx", cache / f"{name}.onnx.json") for name in names}
    downloads = [(voice_url(name), onnx) for name, (onnx, _) in files.items()]
    downloads += [(voice_url(name) + ".json", config) for name, (_, config) in files.items()]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda item: download(*item), downloads))
    return files


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


def pitch_shift(path: Path, factor: float) -> None:
    """Shift pitch and formants by `factor` in place, keeping the duration:
    play it faster/slower (asetrate), then stretch back to length (atempo)."""
    shifted = path.with_name(path.stem + ".pitch.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-af",
         f"asetrate={round(TARGET_RATE * factor)},aresample={TARGET_RATE},"
         f"atempo={1 / factor:.6f}", "-ac", "1", "-sample_fmt", "s16", str(shifted)],
        check=True,
    )
    shifted.replace(path)


def finish(src: Path, out: Path, pitch: float) -> None:
    resample_to_16k(src, out)
    src.unlink()
    if pitch != 1.0:
        pitch_shift(out, pitch)


def check_piper() -> None:
    try:
        import piper  # noqa: F401
    except ImportError:
        sys.exit("piper-tts is missing: python3 -m pip install piper-tts")


_voice_files: dict[str, tuple[Path, Path]] = {}
_loaded_voices: OrderedDict = OrderedDict()


_espeak = ""


def init_piper_worker(voice_files: dict[str, tuple[Path, Path]], espeak: str) -> None:
    global _espeak
    _voice_files.update(voice_files)
    _espeak = espeak


def worker_voice(name: str):
    """Load a voice on first use, keeping the two most recent ones. Jobs are
    sorted by voice, so a worker rarely switches and RAM stays at about two
    models per worker instead of all ten. Sessions are single-threaded so N
    workers use N cores instead of fighting over onnxruntime's thread pool."""
    if name in _loaded_voices:
        _loaded_voices.move_to_end(name)
        return _loaded_voices[name]
    import onnxruntime
    from piper import PiperVoice

    onnx, config = _voice_files[name]
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    voice = PiperVoice.load(str(onnx), str(config))
    voice.session = onnxruntime.InferenceSession(
        str(onnx), sess_options=options, providers=["CPUExecutionProvider"])
    # target-language phonemes for voices of another language; a voice of
    # the language itself keeps its own variant (en-gb stays en-gb)
    if voice.config.espeak_voice.split("-")[0] != _espeak.split("-")[0]:
        voice.config.espeak_voice = _espeak
    _loaded_voices[name] = voice
    while len(_loaded_voices) > 2:
        _loaded_voices.popitem(last=False)
    return voice


def piper_job(job: tuple) -> None:
    from piper import SynthesisConfig

    voice_name, speaker_id, text, length_scale, noise_scale, noise_w_scale, pitch, tmp, out = job
    config = SynthesisConfig(
        speaker_id=speaker_id, length_scale=length_scale,
        noise_scale=noise_scale, noise_w_scale=noise_w_scale)
    with wave.open(str(tmp), "wb") as wav:
        worker_voice(voice_name).synthesize_wav(text, wav, syn_config=config)
    finish(tmp, out, pitch)


def edge_command() -> list[str] | None:
    """Prefer the edge_tts module of this interpreter, so it works from an
    unactivated venv where `edge-tts` is not on PATH."""
    if importlib.util.find_spec("edge_tts"):
        return [sys.executable, "-m", "edge_tts"]
    found = shutil.which("edge-tts")
    return [found] if found else None


def edge_job(job: tuple) -> bool:
    command, text, voice, rate, edge_pitch, pitch, tmp_mp3, out = job
    result = subprocess.run(
        [*command, "--voice", voice, "--text", text,
         f"--rate={rate}", f"--pitch={edge_pitch}", "--write-media", str(tmp_mp3)],
        capture_output=True,
    )
    if result.returncode != 0 or not tmp_mp3.exists():
        return False
    finish(tmp_mp3, out, pitch)
    return True


def say_job(job: tuple) -> bool:
    voice, text, rate, pitch, tmp, out = job
    result = subprocess.run(
        ["say", "-v", voice, "-r", str(rate),
         "-o", str(tmp), "--data-format=LEI16@16000", text],
        capture_output=True,
    )
    if result.returncode != 0 or not tmp.exists():
        return False
    finish(tmp, out, pitch)
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


class Planner:
    """Draws every sample's random settings up front from one seeded RNG."""

    def __init__(self, rng: random.Random, tmp: Path, piper_voices: dict[str, int],
                 multi_speakers: dict[str, int], edge_voices: list[str], say_voice: str,
                 pitch_prob: float, pitch_range: float):
        self.rng, self.tmp = rng, tmp
        self.piper_voices, self.multi_speakers = piper_voices, multi_speakers
        self.edge_voices, self.say_voice = edge_voices, say_voice
        self.pitch_prob, self.pitch_range = pitch_prob, pitch_range
        self.piper, self.edge, self.say = [], [], []

    def pitch(self) -> float:
        if self.rng.random() >= self.pitch_prob:
            return 1.0
        return 2 ** (self.rng.uniform(-self.pitch_range, self.pitch_range) / 12)

    def piper_sample(self, voice: str, speaker: int | None, text: str, out: Path) -> None:
        rng = self.rng
        self.piper.append((voice, speaker, text, rng.uniform(0.65, 1.4), rng.uniform(0.5, 0.8),
                           rng.uniform(0.6, 1.0), self.pitch(),
                           self.tmp / f"piper_{len(self.piper):06d}.wav", out))

    def add(self, out_dir: Path, single: int, multi: int, edge: int, say: int,
            spoken, edge_text) -> None:
        """spoken()/edge_text() return the phrase for Piper+say / edge-tts."""
        rng = self.rng
        voices, weights = list(self.piper_voices), list(self.piper_voices.values())
        for i in range(single):
            self.piper_sample(rng.choices(voices, weights)[0], None, spoken(),
                              out_dir / f"{i:05d}.wav")
        models = list(self.multi_speakers)
        sizes = list(self.multi_speakers.values())
        for i in range(multi):
            model = rng.choices(models, sizes)[0]
            self.piper_sample(model, rng.randrange(self.multi_speakers[model]), spoken(),
                              out_dir / f"multi_{i:05d}.wav")
        for i in range(edge):
            self.edge.append((edge_text(), rng.choice(self.edge_voices),
                              f"{rng.randint(-35, 35):+d}%", f"{rng.randint(-15, 15):+d}Hz",
                              self.pitch(), self.tmp / f"edge_{len(self.edge):06d}.mp3",
                              out_dir / f"edge_{i:05d}.wav"))
        for i in range(say):
            self.say.append((self.say_voice, spoken(), rng.randint(160, 210), self.pitch(),
                             self.tmp / f"say_{len(self.say):06d}.wav",
                             out_dir / f"say_{i:05d}.wav"))


def split_count(total: int, weights: list[int]) -> list[int]:
    """Split `total` in proportion to `weights` (all to the first if all zero)."""
    if sum(weights) == 0:
        weights = [1] + [0] * (len(weights) - 1)
    shares = [total * w // sum(weights) for w in weights]
    shares[weights.index(max(weights))] += total - sum(shares)
    return shares


def variants(phrases: list[str]) -> list[str]:
    """Each phrase as a statement, exclamation and question, unless it
    already ends in punctuation."""
    out = []
    for phrase in phrases:
        phrase = phrase.strip()
        out += [phrase] if phrase[-1] in ".!?" else [phrase + p for p in ".!?"]
    return out


def slug(phrase: str) -> str:
    """"Hey, Computer!" -> hey_computer (file and folder names)."""
    return re.sub(r"\W+", "_", phrase.lower()).strip("_") or "wake_word"


def voice_weights(specs: list[str]) -> dict[str, int]:
    """["en_US-amy-medium:3", "en_US-joe-medium"] -> {name: weight}"""
    weights = {}
    for spec in specs:
        name, _, weight = spec.partition(":")
        weights[name] = int(weight or 1)
    return weights


def zip_folder(folder: Path) -> str:
    return shutil.make_archive(str(folder), "zip", root_dir=folder, base_dir=".")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1000,
                        help="samples from single-speaker Piper voices (default 1000)")
    parser.add_argument("--multi", type=int, default=0,
                        help="samples from multi-speaker Piper models, 1013 speakers")
    parser.add_argument("--edge", type=int, default=0,
                        help="extra samples via edge-tts (needs `pip install edge-tts`)")
    parser.add_argument("--say", type=int, default=0, help="extra samples via macOS `say`")
    parser.add_argument("--phrase", action="append", required=True,
                        help="wake phrase to synthesize (repeatable), e.g. \"Hey Computer\"")
    parser.add_argument("--edge-phrase", action="append", default=[],
                        help="phrase spelling for edge-tts (repeatable; default: --phrase)")
    parser.add_argument("--language", default="en",
                        help=f"voice preset: {', '.join(LANGUAGES)} (default en); any other "
                             "language needs the voice options below")
    parser.add_argument("--piper-voice", action="append", default=[], metavar="NAME[:WEIGHT]",
                        help="single-speaker Piper voice (repeatable; replaces the preset's)")
    parser.add_argument("--multi-voice", action="append", default=[], metavar="NAME",
                        help="multi-speaker Piper model (repeatable; replaces the preset's)")
    parser.add_argument("--edge-voice", action="append", default=[],
                        help="edge-tts voice, e.g. en-US-GuyNeural (repeatable; replaces the preset's)")
    parser.add_argument("--say-voice", default=None, help="macOS say voice (default: the preset's)")
    parser.add_argument("--espeak", default=None,
                        help="espeak-ng voice for Piper's phonemes, e.g. en-us (default: the preset's)")
    parser.add_argument("--pitch-prob", type=float, default=0.3,
                        help="share of synthesized samples to pitch/formant shift (default 0.3)")
    parser.add_argument("--pitch-range", type=float, default=4.0,
                        help="maximum shift in semitones, up or down (default 4)")
    parser.add_argument("--negatives", type=int, default=0,
                        help="negative samples, split across Piper/multi/edge like the positives")
    parser.add_argument("--negative-phrase", action="append", default=[],
                        help="phrase the model must not wake on (repeatable)")
    parser.add_argument("--negatives-out", type=Path, default=None,
                        help="negative sample folder (default: <--out>_negatives)")
    parser.add_argument("--extra-wavs", type=Path, default=None,
                        help="folder of your own recordings to fold in")
    parser.add_argument("--out", type=Path, default=None,
                        help="sample folder (default: samples/<first phrase>, e.g. samples/hey_computer)")
    parser.add_argument("--voice-cache", type=Path, default=Path("samples/voices"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="parallel Piper worker processes (default: all CPU cores)")
    parser.add_argument("--edge-jobs", type=int, default=4,
                        help="parallel edge-tts requests (default 4; more risks throttling)")
    args = parser.parse_args()

    preset = LANGUAGES.get(args.language, {})
    piper_voices = voice_weights(args.piper_voice) or preset.get("piper", {})
    multi_voices = args.multi_voice or preset.get("multi", [])
    edge_voices = args.edge_voice or preset.get("edge", [])
    say_voice = args.say_voice or preset.get("say")
    espeak = args.espeak or preset.get("espeak") or args.language
    for wanted, have, option in [(args.count, piper_voices, "--piper-voice"),
                                 (args.multi, multi_voices, "--multi-voice"),
                                 (args.edge, edge_voices, "--edge-voice"),
                                 (args.say, say_voice, "--say-voice")]:
        if wanted and not have:
            sys.exit(f"no preset for --language {args.language}: pass {option}")
    if args.out is None:
        args.out = Path("samples") / slug(args.phrase[0])
    if args.negatives_out is None:
        args.negatives_out = args.out.with_name(args.out.name + "_negatives")
    print(f"phrases: {', '.join(args.phrase)}  language {args.language}, phonemes {espeak}")

    if args.negatives and not args.negative_phrase:
        sys.exit("--negatives needs at least one --negative-phrase")
    if args.pitch_prob > 0 and not shutil.which("ffmpeg"):
        print("pitch shifting needs ffmpeg - skipping it")
        args.pitch_prob = 0.0
    neg_single, neg_multi, neg_edge = (
        split_count(args.negatives, [args.count, args.multi, args.edge])
        if args.negatives else (0, 0, 0))

    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out.parent / "tmp"
    tmp.mkdir(exist_ok=True)

    needed = []
    if args.count or neg_single:
        needed += list(piper_voices)
    if args.multi or neg_multi:
        needed += multi_voices
    voice_files = {}
    if needed:
        check_piper()
        voice_files = ensure_voices(needed, args.voice_cache)
    multi_speakers = {name: json.loads(voice_files[name][1].read_text())["num_speakers"]
                      for name in multi_voices if name in voice_files}

    rng = random.Random(args.seed)
    plan = Planner(rng, tmp, piper_voices, multi_speakers, edge_voices, say_voice,
                   args.pitch_prob, args.pitch_range)
    spoken = variants(args.phrase)
    edge_spoken = variants(args.edge_phrase) if args.edge_phrase else spoken
    plan.add(args.out, args.count, args.multi, args.edge, args.say,
             spoken=lambda: rng.choice(spoken),
             edge_text=lambda: rng.choice(edge_spoken))
    if args.negatives:
        args.negatives_out.mkdir(parents=True, exist_ok=True)
        plan.add(args.negatives_out, neg_single, neg_multi, neg_edge, 0,
                 spoken=lambda: rng.choice(args.negative_phrase),
                 edge_text=lambda: rng.choice(args.negative_phrase))

    if plan.piper:
        jobs = max(1, min(args.jobs, len(plan.piper)))
        print(f"synthesizing {len(plan.piper)} Piper samples with {jobs} worker(s)")
        piper_jobs = sorted(plan.piper, key=lambda job: job[0])  # group by voice
        with ProcessPoolExecutor(jobs, initializer=init_piper_worker,
                                 initargs=(voice_files, espeak)) as pool:
            for done, _ in enumerate(pool.map(piper_job, piper_jobs, chunksize=8), 1):
                if done % 200 == 0 or done == len(piper_jobs):
                    print(f"{done}/{len(piper_jobs)} Piper samples")

    if plan.edge:
        command = edge_command()
        if not command:
            print("edge-tts not usable - install it with `python3 -m pip install edge-tts`")
        else:
            with ThreadPoolExecutor(max(1, args.edge_jobs)) as pool:
                ok = sum(pool.map(edge_job, [(command, *job) for job in plan.edge]))
            print(f"{ok}/{len(plan.edge)} edge-tts samples")
            if ok < len(plan.edge):
                print("some edge-tts requests failed (network or throttling); "
                      "try a lower --edge-jobs")

    if plan.say:
        if platform.system() != "Darwin":
            print("say is macOS only - skipping --say")
        else:
            with ThreadPoolExecutor(os.cpu_count() or 1) as pool:
                ok = sum(pool.map(say_job, plan.say))
            print(f"{ok}/{len(plan.say)} say samples")
            if ok == 0:
                print(f"say -v {say_voice} not usable - add the voice in System Settings "
                      "(Accessibility > Spoken Content > System voice > Manage Voices)")

    if args.extra_wavs:
        folded = fold_in_extra_wavs(args.extra_wavs, args.out)
        print(f"folded in {folded} real recording(s) from {args.extra_wavs}")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(list(args.out.glob('*.wav')))} positive samples in {args.out}, "
          f"zipped to {zip_folder(args.out)}")
    if args.negatives:
        print(f"{len(list(args.negatives_out.glob('*.wav')))} negative samples in "
              f"{args.negatives_out}, zipped to {zip_folder(args.negatives_out)}")
    print(f"train with: python3 training/train_wake_word.py --wake-word \"{args.phrase[0]}\" "
          f"--language {args.language} --samples {args.out}.zip"
          + (f" --negatives {args.negatives_out}.zip" if args.negatives else ""))


if __name__ == "__main__":
    main()
