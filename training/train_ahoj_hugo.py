#!/usr/bin/env python3
"""Train an "Ahoj Hugo" microWakeWord model locally.

Local equivalent of training/train_ahoj_hugo.ipynb. Point it at the samples
produced by training/generate_samples.py (a directory or the zip) and it does
the rest: augmentation data, negative datasets, spectrogram features,
training, quantization, manifest.

    python3 -m pip install 'datasets<4'
    git clone https://github.com/kahrendt/microWakeWord
    python3 -m pip install -e ./microWakeWord
    python3 training/train_ahoj_hugo.py --samples samples/ahoj_hugo

Without a GPU this is impractically slow - use the Colab notebook instead,
or expect days of training on CPU.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

NEGATIVE_URL = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
NEGATIVES = ["dinner_party.zip", "dinner_party_eval.zip",
             "no_speech.zip", "speech.zip"]
AUDIOSET_SHARD = ("https://huggingface.co/datasets/agkphysics/AudioSet"
                  "/resolve/main/data/bal_train/00.parquet")
FMA_URL = "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"


def need(module: str, hint: str) -> None:
    try:
        __import__(module)
    except ImportError:
        sys.exit(f"missing {module}: {hint}")


def check_environment() -> None:
    need("microwakeword",
         "git clone https://github.com/kahrendt/microWakeWord "
         "&& python3 -m pip install -e ./microWakeWord")
    need("datasets", "python3 -m pip install 'datasets<4'")
    import datasets
    if int(datasets.__version__.split(".")[0]) >= 4:
        sys.exit(f"datasets {datasets.__version__} breaks audio decoding; "
                 "install 'datasets<4'")


def unpack_samples(samples: Path, clips_dir: Path) -> None:
    if clips_dir.exists() and any(clips_dir.glob("*.wav")):
        print(f"using existing {clips_dir}")
        return
    clips_dir.mkdir(parents=True, exist_ok=True)
    if samples.is_file() and samples.suffix == ".zip":
        with zipfile.ZipFile(samples) as archive:
            archive.extractall(clips_dir)
    else:
        for wav in samples.glob("*.wav"):
            shutil.copy(wav, clips_dir / wav.name)
    count = len(list(clips_dir.glob("*.wav")))
    print(f"{count} positive samples in {clips_dir}")
    if count == 0:
        sys.exit(f"no WAVs found in {samples}")


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def write_wav(path: Path, rate: int, samples) -> None:
    import numpy as np
    import scipy.io
    scipy.io.wavfile.write(path, rate, (np.asarray(samples) * 32767).astype(np.int16))


def download_rirs(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.glob("*.wav")):
        return
    import datasets
    from tqdm import tqdm
    out_dir.mkdir(parents=True, exist_ok=True)
    rirs = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train", streaming=True)
    for row in tqdm(rirs, desc="MIT RIRs"):
        name = row['audio']['path'].split('/')[-1]
        write_wav(out_dir / name, 16000, row['audio']['array'])


def download_audioset(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.glob("*.wav")):
        return
    import datasets
    from tqdm import tqdm
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        shard = datasets.load_dataset(
            "parquet", data_files=AUDIOSET_SHARD, split="train",
            streaming=True).cast_column("audio", datasets.Audio(sampling_rate=16000))
        for row in tqdm(shard, desc="AudioSet"):
            name = row['audio']['path'].split('/')[-1].rsplit(".", 1)[0] + ".wav"
            write_wav(out_dir / name, 16000, row['audio']['array'])
    except Exception as error:
        print(f"AudioSet unavailable ({error}); continuing with FMA only")


def download_fma(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.glob("*.wav")):
        return
    import datasets
    from tqdm import tqdm
    out_dir.mkdir(parents=True, exist_ok=True)
    download(FMA_URL, Path("fma/fma_xs.zip"))
    with zipfile.ZipFile("fma/fma_xs.zip") as archive:
        archive.extractall("fma")
    files = datasets.Dataset.from_dict(
        {"audio": [str(i) for i in Path("fma/fma_small").glob("**/*.mp3")]})
    files = files.cast_column("audio", datasets.Audio(sampling_rate=16000))
    for row in tqdm(files, desc="FMA"):
        name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
        write_wav(out_dir / name, 16000, row['audio']['array'])


def download_negatives(out_dir: Path) -> None:
    if out_dir.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in NEGATIVES:
        download(NEGATIVE_URL + name, out_dir / name)
        with zipfile.ZipFile(out_dir / name) as archive:
            archive.extractall(out_dir)


def build_features(clips_dir: Path) -> None:
    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    backgrounds = [d for d in ["fma_16k", "audioset_16k"]
                   if Path(d).is_dir() and any(Path(d).iterdir())]
    if not backgrounds:
        sys.exit("no background audio found; run with --download-data first")

    clips = Clips(input_directory=str(clips_dir), file_pattern="*.wav",
                  max_clip_duration_s=None, remove_silence=False,
                  random_split_seed=10, split_count=0.1)
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1,
            "PitchShift": 0.1, "BandStopFilter": 0.1, "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75, "Gain": 1.0, "RIR": 0.5,
        },
        impulse_paths=["mit_rirs"], background_paths=backgrounds,
        background_min_snr_db=-5, background_max_snr_db=10,
        min_jitter_s=0.195, max_jitter_s=0.205)

    output_dir = Path("generated_augmented_features")
    for split in ["training", "validation", "testing"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        if split == "training":
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter,
                slide_frames=10, step_ms=10)
            split_name, repeat = "train", 2
        elif split == "validation":
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter,
                slide_frames=10, step_ms=10)
            split_name, repeat = "validation", 1
        else:
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter,
                slide_frames=1, step_ms=10)
            split_name, repeat = "test", 1
        RaggedMmap.from_generator(
            out_dir=str(output_dir / split / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name, repeat=repeat),
            batch_size=1)


def write_config() -> None:
    import yaml
    config = {
        "window_step_ms": 10,
        "train_dir": "trained_models/wakeword",
        "features": [
            {"features_dir": "generated_augmented_features",
             "sampling_weight": 2.0, "penalty_weight": 1.0, "truth": True,
             "truncation_strategy": "truncate_start", "type": "mmap"},
            {"features_dir": "negative_datasets/speech",
             "sampling_weight": 10.0, "penalty_weight": 1.0, "truth": False,
             "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": "negative_datasets/dinner_party",
             "sampling_weight": 10.0, "penalty_weight": 1.0, "truth": False,
             "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": "negative_datasets/no_speech",
             "sampling_weight": 5.0, "penalty_weight": 1.0, "truth": False,
             "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": "negative_datasets/dinner_party_eval",
             "sampling_weight": 0.0, "penalty_weight": 1.0, "truth": False,
             "truncation_strategy": "split", "type": "mmap"},
        ],
        "training_steps": [15000, 5000],
        "positive_class_weight": [1, 1],
        "negative_class_weight": [20, 20],
        "learning_rates": [0.001, 0.0001],
        "batch_size": 128,
        "time_mask_max_size": [0, 10],
        "time_mask_count": [0, 2],
        "freq_mask_max_size": [0, 4],
        "freq_mask_count": [0, 1],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    with open("training_parameters.yaml", "w") as file:
        yaml.dump(config, file)


def train() -> None:
    subprocess.run(
        [sys.executable, "-m", "microwakeword.model_train_eval",
         "--training_config=training_parameters.yaml",
         "--train", "1", "--restore_checkpoint", "1",
         "--test_tf_nonstreaming", "0",
         "--test_tflite_nonstreaming", "0",
         "--test_tflite_nonstreaming_quantized", "0",
         "--test_tflite_streaming", "0",
         "--test_tflite_streaming_quantized", "1",
         "--use_weights", "best_weights",
         "mixednet",
         "--pointwise_filters", "64,64,64,64",
         "--repeat_in_block", "1, 1, 1, 1",
         "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
         "--residual_connection", "0,0,0,0",
         "--first_conv_filters", "32",
         "--first_conv_kernel_size", "5",
         "--stride", "3"],
        check=True)


def package(out_dir: Path) -> None:
    produced = sorted(Path("trained_models/wakeword").rglob("*.tflite"))
    if not produced:
        sys.exit("training finished but produced no .tflite")
    quantized = [p for p in produced if "quant" in str(p)]
    tflite = (quantized or produced)[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(tflite, out_dir / "ahoj_hugo.tflite")
    manifest = {
        "type": "micro",
        "wake_word": "Ahoj Hugo",
        "author": "skoky",
        "website": "https://github.com/skoky/havoice",
        "model": "ahoj_hugo.tflite",
        "trained_languages": ["cs"],
        "version": 2,
        "micro": {
            "probability_cutoff": 0.97,
            "feature_step_size": 10,
            "sliding_window_size": 5,
            "tensor_arena_size": 30000,
            "minimum_esphome_version": "2024.7.0",
        },
    }
    with open(out_dir / "ahoj_hugo.json", "w") as file:
        json.dump(manifest, file, indent=2)
    print(f"model + manifest written to {out_dir}/")
    print("verify with: cargo run --example wake_word -- verify "
          f"{out_dir}/ahoj_hugo.tflite --wake-word 'Ahoj Hugo'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=Path("samples/ahoj_hugo"),
                        help="sample directory or zip from generate_samples.py")
    parser.add_argument("--out", type=Path, default=Path("ahoj_hugo_model"),
                        help="where to write ahoj_hugo.tflite/.json")
    parser.add_argument("--workdir", type=Path, default=Path("training_run"),
                        help="scratch directory for datasets and features")
    parser.add_argument("--skip-download-data", action="store_true",
                        help="reuse previously downloaded augmentation/negative data")
    args = parser.parse_args()

    check_environment()
    samples = args.samples.resolve()
    out_dir = args.out.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)
    unpack_samples(samples, Path("generated_samples"))

    if not args.skip_download_data:
        download_rirs(Path("mit_rirs"))
        download_audioset(Path("audioset_16k"))
        download_fma(Path("fma_16k"))
        download_negatives(Path("negative_datasets"))
    build_features(Path("generated_samples"))
    write_config()
    train()
    package(out_dir)


if __name__ == "__main__":
    main()
