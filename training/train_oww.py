#!/usr/bin/env python3
"""Train an openWakeWord classifier for any wake word, for a streaming runtime.

The model is for a runtime that runs openWakeWord's three-model chain one
80 ms chunk at a time (e.g. with tract in Rust, as examples/test_wavs.rs does):

    16 kHz audio, floats in [-1, 1], 1280-sample chunks (80 ms)
      -> melspectrogram.onnx on each chunk alone: [5, 32], then v / 10 + 2
      -> ring of the last 16 chunks (80 frames); frames 4..80 -> embedding_model.onnx
      -> ring of the last 16 embeddings: [1, 16, 96]
      -> <name>.onnx -> [1, 1] probability

Only the last model is wake-word specific. This script trains it and exports
an ONNX with openWakeWord's DNN architecture (32-unit layers, layer norm
decomposed into basic ops, opset 13).

The features are computed the way such a runtime computes them, not the way
the openWakeWord Python package does: running the spectrogram model on each
chunk without the 480-sample overlap gives 5 frames per 80 ms instead of 8. A
classifier trained on openWakeWord's own features (and its precomputed
negative sets) sees differently scaled input there; one trained here sees the
same numbers as at runtime.

Training data:
  positives  samples from generate_samples.py, each mixed into
             music/speech/noise backgrounds with room reverb, at several
             random positions; windows from 0 to 400 ms after the phrase end
             are positive.
  negatives  FLEURS read speech in --language (CC-BY-4.0, ~1-2 GB download),
             FMA music from training_run/ (downloaded by train_wake_word.py or
             here), FLEURS speech over music, synthetic noise, the first half
             of every positive phrase (the first word alone must not fire),
             and optional generated negatives (--negatives).

Validation uses the runtime's decision rule (threshold, 2+ positive
detections in the last 12 chunks, 2 s refractory): recall on held-out
positives, false wake-ups per hour on FLEURS dev speech, and, with
--test-positive / --test-negative, your own recordings.

    python3 training/train_oww.py --samples samples/hey_computer.zip
    python3 training/train_oww.py --samples samples/hey_computer.zip --language de \\
        --negatives samples/hey_computer_negatives.zip --test-positive my_recordings

Writes <name>_model/<name>.onnx (--out), named after the samples.
"""

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import zipfile
from collections import deque
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SR = 16000
CHUNK = 1280                  # samples per streaming step (80 ms)
MEL_FRAMES = 5                # frames melspectrogram.onnx returns for one chunk
CONTEXT_CHUNKS = 31           # 16 embeddings, each over 16 chunks
FEATURE_MODELS = {            # openWakeWord v0.5.1
    "melspectrogram.onnx":
        "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx":
        "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
}
OWW_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{code}/audio/"
# FLEURS config for a language code; others are passed with --fleurs
FLEURS_CODES = {
    "cs": "cs_cz", "da": "da_dk", "de": "de_de", "en": "en_us", "es": "es_419",
    "fi": "fi_fi", "fr": "fr_fr", "hu": "hu_hu", "it": "it_it", "nl": "nl_nl",
    "pl": "pl_pl", "pt": "pt_br", "ru": "ru_ru", "sk": "sk_sk", "sv": "sv_se",
    "uk": "uk_ua",
}
# Detection rule: the probabilities of the last 12 chunks, more than
# min_positive above the threshold, their average above it, 2 s refractory
DETECTION_BUFFER = 12
REFRACTORY_CHUNKS = 2000 // 80
DEFAULT_MIN_POSITIVE = 2


# ---------------------------------------------------------------- audio I/O

def read_wav_bytes(data: bytes) -> np.ndarray:
    """WAV (16-bit PCM, float, ...) -> mono float32 in [-1, 1] at 16 kHz,
    scaled like an i16 -> f32 conversion (x / 32768)."""
    import soundfile
    audio, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SR:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, SR, rate).astype(np.float32)
    return audio


def read_wav(path: Path) -> np.ndarray:
    return read_wav_bytes(path.read_bytes())


def load_clip_set(source: Path) -> dict[str, np.ndarray]:
    """All WAVs of a directory or zip from generate_samples.py."""
    clips = {}
    if source.is_file() and source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith(".wav"):
                    clips[Path(name).name] = read_wav_bytes(archive.read(name))
    else:
        for path in sorted(source.glob("*.wav")):
            clips[path.name] = read_wav(path)
    if not clips:
        sys.exit(f"no WAVs found in {source}")
    return clips


def trim(audio: np.ndarray, top_db: float = 35.0) -> np.ndarray:
    """Cut leading/trailing silence so the clip end is the phrase end."""
    frame = 160
    n = len(audio) // frame
    if n == 0:
        return audio
    energy = 10 * np.log10((audio[:n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-10)
    loud = np.nonzero(energy > energy.max() - top_db)[0]
    return audio[loud[0] * frame:(loud[-1] + 1) * frame]


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-9)


# ---------------------------------------------------------- streaming features

class StreamingFeatures:
    """openWakeWord features computed the way a streaming runtime does: one
    melspectrogram call per 1280-sample chunk, zero-initialized rings of 16
    mel chunks and 16 embeddings."""

    def __init__(self, models_dir: Path, threads: int):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self.mel = ort.InferenceSession(str(models_dir / "melspectrogram.onnx"), options,
                                        providers=providers)
        self.emb = ort.InferenceSession(str(models_dir / "embedding_model.onnx"), options,
                                        providers=providers)

    def embeddings(self, audio: np.ndarray) -> np.ndarray:
        """[n_chunks, 96]: embedding i is what the runtime computes after
        receiving chunk i of this stream."""
        n = len(audio) // CHUNK
        if n == 0:
            return np.zeros((0, 96), np.float32)
        chunks = audio[:n * CHUNK].reshape(n, CHUNK).astype(np.float32)
        mels = np.concatenate([
            self.mel.run(None, {"input": chunks[i:i + 2048]})[0]
            for i in range(0, n, 2048)]).reshape(n, MEL_FRAMES, 32)
        mels = mels / 10.0 + 2.0
        frames = np.concatenate([np.zeros((15 * MEL_FRAMES, 32), np.float32),
                                 mels.reshape(-1, 32)])
        out = np.empty((n, 96), np.float32)
        for start in range(0, n, 1024):
            stop = min(n, start + 1024)
            batch = np.stack([frames[i * MEL_FRAMES + 4:i * MEL_FRAMES + 80]
                              for i in range(start, stop)])[..., None]
            out[start:stop] = self.emb.run(None, {"input_1": batch})[0].reshape(-1, 96)
        return out

    @staticmethod
    def windows(emb: np.ndarray) -> np.ndarray:
        """[n_chunks, 16, 96] classifier inputs (zeros before the stream start)."""
        padded = np.concatenate([np.zeros((15, 96), np.float32), emb])
        view = np.lib.stride_tricks.sliding_window_view(padded, 16, axis=0)
        return view.transpose(0, 2, 1)


# ----------------------------------------------------------------- data sources

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_models(local: Path | None, cache: Path) -> Path:
    """The feature models in `local` if they are the expected ones, else
    openWakeWord's release (the same files, checked by hash)."""
    if local:
        if all((local / name).is_file() and sha256(local / name) == digest
               for name, digest in FEATURE_MODELS.items()):
            return local
        print(f"note: {local} doesn't hold the expected feature models; "
              "using openWakeWord v0.5.1")
    import train_wake_word as mww
    for name, digest in FEATURE_MODELS.items():
        mww.download(OWW_RELEASE + name, cache / name)
        if sha256(cache / name) != digest:
            sys.exit(f"{cache / name} has an unexpected hash; delete it and retry")
    return cache


def fleurs(code: str, split: str, out_dir: Path) -> list[Path]:
    """FLEURS read speech (config `code`, e.g. en_us), extracted to
    out_dir/<split>/*.wav."""
    import train_wake_word as mww
    target = out_dir / split
    if not (target.is_dir() and any(target.glob("*.wav"))):
        archive = out_dir / f"{split}.tar.gz"
        mww.download(FLEURS.format(code=code) + f"{split}.tar.gz", archive)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".wav"):
                    member.name = Path(member.name).name
                    tar.extract(member, target, filter="data")
        archive.unlink()
    return sorted(target.glob("*.wav"))


def background_audio() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Music (FMA, AudioSet if present) and room impulse responses from the
    microWakeWord workdir, downloading them if needed."""
    import train_wake_word as mww
    mww.patch_datasets_pickler()
    mww.download_rirs(Path("mit_rirs"))
    mww.download_fma(Path("fma_16k"))
    music = [read_wav(p) for d in ["fma_16k", "audioset_16k"] if Path(d).is_dir()
             for p in sorted(Path(d).glob("*.wav"))]
    rirs = [read_wav(p) for p in sorted(Path("mit_rirs").glob("*.wav"))]
    return music, rirs


def synthetic_noise(rng: np.random.Generator, length: int) -> np.ndarray:
    white = rng.standard_normal(length).astype(np.float32)
    kind = rng.integers(3)
    if kind == 0:
        return white
    spectrum = np.fft.rfft(white)
    freqs = np.maximum(np.fft.rfftfreq(length), 1.0 / length)
    spectrum /= np.sqrt(freqs) if kind == 1 else freqs     # pink or brown
    noise = np.fft.irfft(spectrum, length).astype(np.float32)
    return noise / (rms(noise) / rms(white))


# ----------------------------------------------------------------- augmentation

class Mixer:
    def __init__(self, rng: np.random.Generator, music: list[np.ndarray],
                 speech: list[np.ndarray], rirs: list[np.ndarray]):
        self.rng, self.music, self.speech, self.rirs = rng, music, speech, rirs

    def segment(self, pool: list[np.ndarray], length: int) -> np.ndarray:
        source = pool[self.rng.integers(len(pool))]
        if len(source) < length:
            source = np.tile(source, length // len(source) + 1)
        start = self.rng.integers(len(source) - length + 1)
        return source[start:start + length]

    def background(self, length: int) -> np.ndarray:
        choice = self.rng.random()
        if choice < 0.4 and self.music:
            return self.segment(self.music, length)
        if choice < 0.65 and self.speech:
            return self.segment(self.speech, length)
        return synthetic_noise(self.rng, length)

    def reverb(self, clip: np.ndarray) -> np.ndarray:
        from scipy.signal import fftconvolve
        rir = self.rirs[self.rng.integers(len(self.rirs))]
        wet = fftconvolve(clip, rir)[:len(clip) + SR // 4].astype(np.float32)
        return wet * (np.abs(clip).max() / (np.abs(wet).max() + 1e-9))

    def place(self, clip: np.ndarray, pre_s: float, post_s: float) -> tuple[np.ndarray, int, int]:
        """Mix a phrase into a background: reverb (50%), speech level -40 to
        -12 dBFS, background at 0-25 dB SNR (80%) or barely audible.
        Returns (audio, phrase start, phrase end)."""
        rng = self.rng
        dry = trim(clip)
        speech = self.reverb(dry) if self.rirs and rng.random() < 0.5 else dry
        speech = speech * (10 ** (rng.uniform(-40, -12) / 20) / rms(speech))
        pre = int(pre_s * SR) + int(rng.integers(CHUNK))  # random alignment to chunks
        length = pre + len(speech) + int(post_s * SR)
        background = self.background(length)
        snr = rng.uniform(0, 25) if rng.random() < 0.8 else rng.uniform(40, 60)
        audio = background * (rms(speech) / rms(background) / 10 ** (snr / 20))
        audio[pre:pre + len(speech)] += speech
        peak = np.abs(audio).max()
        if peak > 0.99:
            audio *= 0.99 / peak
        return audio.astype(np.float32), pre, pre + len(dry)  # reverb tail not counted


def phrase_windows(features: StreamingFeatures, mixer: Mixer, clip: np.ndarray,
                   positive: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classifier windows for one phrase in a background.

    Returns (windows from 0-400 ms after the phrase end, windows over its first
    half, windows that end before it starts). The last are negatives with the
    exact background of the positives, including the zero-filled start-up
    windows the runtime sees right after it starts listening, so the model
    can't key on the background instead of the phrase. For a negative phrase
    every window is returned as the first element."""
    context = CONTEXT_CHUNKS * CHUNK / SR
    audio, start, end = mixer.place(clip, pre_s=context + 0.05, post_s=0.5)
    windows = features.windows(features.embeddings(audio))
    ends = (np.arange(len(windows)) + 1) * CHUNK          # last sample each window saw
    full = np.arange(len(windows)) >= CONTEXT_CHUNKS - 1
    if positive:
        after = full & (ends >= end) & (ends < end + int(0.4 * SR))
        first_half = full & (ends >= start + int(0.15 * SR)) & (ends <= start + (end - start) // 2)
        return windows[after], windows[first_half], windows[ends < start]
    return windows, windows[:0], windows[:0]


def stream_windows(features: StreamingFeatures, audio: np.ndarray, stride: int) -> np.ndarray:
    windows = features.windows(features.embeddings(audio))
    return windows[CONTEXT_CHUNKS - 1::stride].astype(np.float16)


def concat(clips: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Join clips into one stream at varied levels (-40 to -12 dBFS)."""
    parts = [c * (10 ** (rng.uniform(-40, -12) / 20) / rms(c)) for c in clips if len(c)]
    return np.clip(np.concatenate(parts), -1, 1).astype(np.float32)


# ------------------------------------------------------------- model + export

def build_model(layer_size: int):
    import keras
    inputs = keras.Input((16, 96))
    x = keras.layers.Flatten()(inputs)
    for _ in range(2):  # openWakeWord's Net: first layer + one block
        x = keras.layers.Dense(layer_size)(x)
        x = keras.layers.LayerNormalization(epsilon=1e-5)(x)
        x = keras.layers.ReLU()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


def numpy_weights(model) -> list[np.ndarray]:
    return [w.astype(np.float32) for w in model.get_weights()]


def forward(weights: list[np.ndarray], x: np.ndarray, batch: int = 65536) -> np.ndarray:
    """The classifier in numpy (same math as the exported ONNX)."""
    w1, b1, g1, be1, w2, b2, g2, be2, w3, b3 = weights
    out = np.empty(len(x), np.float32)
    for start in range(0, len(x), batch):
        h = x[start:start + batch].reshape(-1, 16 * 96).astype(np.float32)
        for w, b, g, be in ((w1, b1, g1, be1), (w2, b2, g2, be2)):
            h = h @ w + b
            mean = h.mean(axis=1, keepdims=True)
            d = h - mean
            h = np.maximum(d / np.sqrt((d ** 2).mean(axis=1, keepdims=True) + 1e-5) * g + be, 0)
        z = (h @ w3 + b3)[:, 0]
        out[start:start + batch] = 1 / (1 + np.exp(-z))
    return out


def export_onnx(weights: list[np.ndarray], path: Path) -> None:
    """Same graph as torch.onnx.export of openWakeWord's Net (opset 13):
    Flatten, Gemm, layer norm as ReduceMean/Sub/Pow/Sqrt/Div/Mul/Add, Relu,
    Sigmoid: basic operators every ONNX runtime (tract included) has."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, inits = [], []

    def const(name, value):
        inits.append(numpy_helper.from_array(np.asarray(value, np.float32), name))
        return name

    def node(op, inputs, name, **attrs):
        nodes.append(helper.make_node(op, inputs, [name], name=name, **attrs))
        return name

    w1, b1, g1, be1, w2, b2, g2, be2, w3, b3 = weights
    x = node("Flatten", ["input"], "flat", axis=1)
    pow2, eps = const("two", 2.0), const("eps", 1e-5)
    for i, (w, b, g, be) in enumerate(((w1, b1, g1, be1), (w2, b2, g2, be2)), 1):
        x = node("Gemm", [x, const(f"w{i}", w.T), const(f"b{i}", b)], f"gemm{i}", transB=1)
        mean = node("ReduceMean", [x], f"mean{i}", axes=[-1], keepdims=1)
        d = node("Sub", [x, mean], f"centered{i}")
        var = node("ReduceMean", [node("Pow", [d, pow2], f"sq{i}")], f"var{i}",
                   axes=[-1], keepdims=1)
        std = node("Sqrt", [node("Add", [var, eps], f"vareps{i}")], f"std{i}")
        x = node("Div", [d, std], f"norm{i}")
        x = node("Add", [node("Mul", [x, const(f"gamma{i}", g)], f"scaled{i}"),
                         const(f"beta{i}", be)], f"ln{i}")
        x = node("Relu", [x], f"relu{i}")
    x = node("Gemm", [x, const("w3", w3.T), const("b3", b3)], "logit", transB=1)
    node("Sigmoid", [x], "output")
    graph = helper.make_graph(
        nodes, path.stem,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 96])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="havoice train_oww.py")
    model.ir_version = 7
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


# ------------------------------------------------------------------ evaluation

def detections(probs: np.ndarray, threshold: float, min_positive: int
               ) -> list[tuple[int, float]]:
    """(chunk index, average probability) wherever the runtime would fire:
    the average of the hits among the last 12 chunks."""
    buffer: deque = deque(maxlen=DETECTION_BUFFER)
    fired, last = [], -REFRACTORY_CHUNKS - 1
    for i, p in enumerate(probs):
        buffer.append(p)
        hits = [v for v in buffer if v > threshold]
        average = sum(hits) / len(hits) if len(hits) > min_positive else 0.0
        if average > threshold and i - last > REFRACTORY_CHUNKS:
            fired.append((i, float(average)))
            last = i
            buffer.clear()
    return fired


def stream_probs(weights, features: StreamingFeatures, audio: np.ndarray) -> np.ndarray:
    return forward(weights, features.windows(features.embeddings(audio)))


class Evaluator:
    def __init__(self, features, val_positive_streams, val_negative_audio, recordings,
                 threshold):
        self.features, self.threshold = features, threshold
        # Precompute windows once; only the classifier changes between evaluations.
        # (windows, first window index that ends after the phrase starts)
        self.pos = [(features.windows(features.embeddings(a))[CONTEXT_CHUNKS - 1:],
                     max(0, start // CHUNK - (CONTEXT_CHUNKS - 1)))
                    for a, start in val_positive_streams]
        self.neg = features.windows(features.embeddings(val_negative_audio))[CONTEXT_CHUNKS - 1:]
        self.neg_hours = len(val_negative_audio) / SR / 3600
        self.tests = [(name, expected, features.windows(features.embeddings(a)))
                      for name, expected, a in recordings]

    def __call__(self, weights) -> dict:
        t, k = self.threshold, DEFAULT_MIN_POSITIVE
        recall = np.mean([any(i >= first for i, _ in detections(forward(weights, w), t, k))
                          for w, first in self.pos])
        false = len(detections(forward(weights, self.neg), t, k))
        result = {"recall": float(recall), "false_per_hour": false / self.neg_hours}
        if self.tests:
            result["recordings"] = self.recordings(weights)
        return result

    def recordings(self, weights) -> list[tuple[str, bool, bool, float]]:
        """Your recordings, with the same detection rule as validation."""
        rows = []
        for name, expected, windows in self.tests:
            probs = forward(weights, windows)
            fired = bool(detections(probs, self.threshold, DEFAULT_MIN_POSITIVE))
            rows.append((name, expected, fired, float(probs.max())))
        return rows


# --------------------------------------------------------------------- training

def train(args, pos, neg, val_eval: Evaluator):
    import keras
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build_model(args.layer_size)
    schedule = keras.optimizers.schedules.CosineDecay(args.learning_rate, args.steps, alpha=0.01)
    model.compile(optimizer=keras.optimizers.Adam(schedule), loss="binary_crossentropy")

    n_pos = args.batch_size // 4
    n_neg = args.batch_size - n_pos
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.float32)[:, None]
    best, best_key, history = None, None, []
    start = time.time()
    for step in range(1, args.steps + 1):
        # Negative weight ramps up as in openWakeWord: learn the wake word
        # first, then push false positives down.
        neg_weight = 1 + (args.max_negative_weight - 1) * min(1.0, step / (0.5 * args.steps))
        weights = np.concatenate([np.ones(n_pos), np.full(n_neg, neg_weight)]).astype(np.float32)
        x = np.concatenate([pos[rng.integers(len(pos), size=n_pos)],
                            neg[rng.integers(len(neg), size=n_neg)]]).astype(np.float32)
        loss = model.train_on_batch(x, labels, sample_weight=weights)
        if step % args.eval_every == 0 or step == args.steps:
            result = val_eval(numpy_weights(model))
            result.update(step=step, loss=float(loss), negative_weight=neg_weight)
            history.append(result)
            ok = result["false_per_hour"] <= args.target_false_per_hour
            key = (ok, result["recall"] if ok else -result["false_per_hour"])
            marker = ""
            if best_key is None or key > best_key:
                best, best_key, marker = numpy_weights(model), key, " *best"
            print(f"step {step}: loss {loss:.4f}, recall {result['recall']:.3f}, "
                  f"false wake-ups/h {result['false_per_hour']:.2f} "
                  f"({time.time() - start:.0f}s){marker}", flush=True)
    return best, history


# ------------------------------------------------------------------------ main

def build_dataset(args, features, rng, samples, negative_clips, music, speech, rirs):
    """Positive and negative classifier windows (float16)."""
    mixer = Mixer(rng, music, speech, rirs)
    pos, partial, before, neg_phrases = [], [], [], []
    names = sorted(samples)
    t0 = time.time()
    for round_index in range(args.rounds):
        for i, name in enumerate(names):
            after, first_half, lead_in = phrase_windows(features, mixer, samples[name], True)
            pos.append(after.astype(np.float16))
            partial.append(first_half.astype(np.float16))
            before.append(lead_in.astype(np.float16))
            if (i + 1) % 1000 == 0:
                print(f"  positives round {round_index + 1}: {i + 1}/{len(names)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    for clip in negative_clips:
        for _ in range(args.rounds):
            windows, _, _ = phrase_windows(features, mixer, clip, False)
            neg_phrases.append(windows.astype(np.float16))
    return (np.concatenate(pos), np.concatenate(partial), np.concatenate(before),
            np.concatenate(neg_phrases) if neg_phrases else np.zeros((0, 16, 96), np.float16))


def signature(items: list) -> str:
    """Short hash of JSON-able items (already in a fixed order)."""
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, required=True,
                        help="positive samples (directory or zip) from generate_samples.py")
    parser.add_argument("--negatives", type=Path, default=None,
                        help="optional negative samples from generate_samples.py --negatives")
    parser.add_argument("--language", default="en",
                        help="language of the FLEURS speech negatives (default en)")
    parser.add_argument("--fleurs", default=None,
                        help="FLEURS config instead, e.g. es_419 (default: from --language)")
    parser.add_argument("--test-positive", type=Path, default=None,
                        help="folder of real recordings that must wake the model")
    parser.add_argument("--test-negative", type=Path, default=None,
                        help="folder of real recordings that must not")
    parser.add_argument("--feature-models", type=Path, default=None,
                        help="folder with melspectrogram.onnx and embedding_model.onnx "
                             "(default: download openWakeWord v0.5.1 into <workdir>/oww/models)")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <name>_model/<name>.onnx, <name> from the samples")
    parser.add_argument("--workdir", type=Path, default=Path("training_run"),
                        help="shared with train_wake_word.py (music, RIRs); oww data in oww/")
    parser.add_argument("--rounds", type=int, default=3,
                        help="augmented placements per positive sample (default 3)")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--layer-size", type=int, default=32,
                        help="units per layer (default 32, openWakeWord's)")
    parser.add_argument("--max-negative-weight", type=float, default=300.0,
                        help="final weight of negatives vs positives (default 300)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="the runtime's detection threshold (default 0.5)")
    parser.add_argument("--target-false-per-hour", type=float, default=0.5,
                        help="best checkpoint = highest recall at or below this rate")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--negative-stride", type=int, default=2,
                        help="use every Nth window of long negative audio (default 2)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    samples_path = args.samples.resolve()
    negatives_path = args.negatives.resolve() if args.negatives else None
    if args.out is None:
        name = args.samples.stem if args.samples.suffix == ".zip" else args.samples.name
        args.out = Path(f"{name}_model") / f"{name}.onnx"
    out = args.out.resolve()
    local_models = args.feature_models.expanduser().resolve() if args.feature_models else None
    test_dirs = [(d.expanduser().resolve(), expected)
                 for d, expected in [(args.test_positive, True), (args.test_negative, False)] if d]
    code = args.fleurs or FLEURS_CODES.get(args.language, f"{args.language}_{args.language}")
    language = code.split("_")[0]
    sys.path.insert(0, str(SCRIPT_DIR))
    args.workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)
    oww = Path("oww")
    oww.mkdir(exist_ok=True)

    models = feature_models(local_models, oww / "models")
    print(f"feature models: {models}")
    features = StreamingFeatures(models, args.threads)
    rng = np.random.default_rng(args.seed)

    print("loading samples")
    samples = load_clip_set(samples_path)
    held_out = {n for n in samples
                if int(hashlib.md5(n.encode()).hexdigest(), 16) % 10 == 0}
    train_samples = {n: c for n, c in samples.items() if n not in held_out}
    negative_clips = list(load_clip_set(negatives_path).values()) if negatives_path else []
    print(f"{len(train_samples)} training and {len(held_out)} validation positives, "
          f"{len(negative_clips)} generated negatives")

    print(f"background audio, room impulse responses, FLEURS {code} speech")
    music, rirs = background_audio()
    fleurs_train = [read_wav(p) for p in fleurs(code, "train", oww / f"fleurs_{language}")]
    fleurs_dev = [read_wav(p) for p in fleurs(code, "dev", oww / f"fleurs_{language}")]
    print(f"{len(music)} music clips ({sum(map(len, music)) / SR / 3600:.1f} h), "
          f"{len(rirs)} RIRs, FLEURS {sum(map(len, fleurs_train)) / SR / 3600:.1f} h train, "
          f"{sum(map(len, fleurs_dev)) / SR / 3600:.1f} h dev")

    # Long negative streams don't depend on the samples: cache them.
    neg_cache = oww / f"negatives_{code}_stride{args.negative_stride}_seed{args.seed}.npy"
    if neg_cache.exists():
        stream_neg = np.load(neg_cache)
    else:
        print("computing negative features (cached in training_run/oww/)")
        t0 = time.time()
        stride = args.negative_stride
        speech = concat(fleurs_train, rng)
        mixer = Mixer(rng, music, [], [])
        noisy = speech.copy()
        step = 30 * SR
        for start in range(0, len(noisy), step):  # music under speech, 0-15 dB SNR
            part = noisy[start:start + step]
            bg = mixer.segment(music, len(part)) if music else synthetic_noise(rng, len(part))
            part += bg * (rms(part) / rms(bg) / 10 ** (rng.uniform(0, 15) / 20))
        parts = [stream_windows(features, speech, stride),
                 stream_windows(features, np.clip(noisy, -1, 1), stride)]
        if music:
            parts.append(stream_windows(features, concat(music, rng), stride))
        noise = np.concatenate([synthetic_noise(rng, 30 * SR) * 10 ** (rng.uniform(-50, -20) / 20)
                                for _ in range(60)])
        parts.append(stream_windows(features, np.clip(noise, -1, 1), stride))
        stream_neg = np.concatenate(parts).astype(np.float16)
        np.save(neg_cache, stream_neg)
        print(f"  {len(stream_neg)} windows in {time.time() - t0:.0f}s")

    sig = signature([(n, len(c)) for n, c in train_samples.items()]
                    + [len(c) for c in negative_clips] + [args.rounds, args.seed, "v2"])
    phrase_cache = oww / f"phrases_{sig}.npz"
    if phrase_cache.exists():
        data = np.load(phrase_cache)
        pos, partial, before, neg_phrases = (data["pos"], data["partial"], data["before"],
                                             data["neg"])
    else:
        print(f"computing positive features ({args.rounds} augmented placements per sample)")
        pos, partial, before, neg_phrases = build_dataset(
            args, features, rng, train_samples, negative_clips, music, fleurs_train, rirs)
        np.savez(phrase_cache, pos=pos, partial=partial, before=before, neg=neg_phrases)
    neg = np.concatenate([stream_neg, partial, before, neg_phrases])
    print(f"training windows: {len(pos)} positive, {len(neg)} negative "
          f"({len(partial)} phrase starts, {len(before)} lead-ins, "
          f"{len(neg_phrases)} generated negatives)")

    print("validation set")
    val_rng = np.random.default_rng(args.seed + 1)
    val_mixer = Mixer(val_rng, music, fleurs_dev, rirs)
    val_pos = [val_mixer.place(samples[n], pre_s=3.0, post_s=1.5)[:2] for n in sorted(held_out)]
    val_neg = concat(fleurs_dev, val_rng)
    tests = [(path.name, expected, read_wav(path))
             for folder, expected in test_dirs for path in sorted(folder.glob("*.wav"))]
    evaluator = Evaluator(features, val_pos, val_neg, tests, args.threshold)
    print(f"  {len(val_pos)} positive streams, {evaluator.neg_hours:.2f} h FLEURS dev speech, "
          f"{len(tests)} recordings")

    weights, history = train(args, pos, neg, evaluator)

    export_onnx(weights, out)
    import onnxruntime as ort
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    probe = neg[:64].astype(np.float32)
    onnx_out = np.concatenate([session.run(None, {"input": w[None]})[0][:, 0] for w in probe])
    if not np.allclose(onnx_out, forward(weights, probe), atol=1e-5):
        sys.exit("exported ONNX disagrees with the trained model")

    final = evaluator(weights)
    print(f"\nbest model: recall {final['recall']:.3f} on held-out positives, "
          f"{final['false_per_hour']:.2f} false wake-ups per hour of FLEURS {code} speech "
          f"(threshold {args.threshold})")
    for name, expected, fired, peak in final.get("recordings", []):
        status = "ok  " if fired == expected else "FAIL"
        print(f"  {status} {name}: expected {'detect' if expected else 'silent'}, "
              f"{'detected' if fired else 'silent'} (max probability {peak:.2f})")
    (out.with_suffix(".history.json")).write_text(json.dumps(history, indent=1))
    print(f"\nwrote {out}")
    print(f"test it with: cargo run --release --example test_wavs -- <folder> --onnx {out}")


if __name__ == "__main__":
    main()
