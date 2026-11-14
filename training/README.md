# Training an "Ahoj Hugo" wake word model

The Voice PE only runs [microWakeWord](https://github.com/kahrendt/microWakeWord) models
(int8 streaming TFLite, 40-bin spectrogram input). An openWakeWord model can't be converted
into one; it has to be trained from audio. This directory covers every step, from no data
to a flashed device:

```
generate_samples.py ──► samples/ahoj_hugo.zip ──► train_ahoj_hugo.py ──► ahoj_hugo_model/
   (Piper, edge-tts,        1000+ 16 kHz WAVs       (or the Colab          ahoj_hugo.tflite
    say, your voice)                                 notebook)              ahoj_hugo.json
                                                                                 │
                                         cargo run --example wake_word -- install --flash
```

| file | purpose |
|---|---|
| `generate_samples.py` | Czech positive samples (Piper voices, edge-tts, macOS `say`, real recordings) |
| `train_ahoj_hugo.py` | local training: downloads, augmentation, features, training, quantization, manifest |
| `train_ahoj_hugo.ipynb` | the same pipeline as a Google Colab notebook (free GPU) |

Run every command below **from the repository root**. The scripts use paths relative to it
(`samples/`, `training_run/`, `ahoj_hugo_model/`), and `.gitignore` already excludes the
large ones.

## Where to train

| | Ubuntu + NVIDIA GPU | Ubuntu, CPU only | macOS (Apple Silicon / Intel) | Google Colab |
|---|---|---|---|---|
| Sample generation | yes | yes | yes, also `say` voices | upload the zip |
| Training | **fast** | slow | slow (CPU only, see below) | fast (T4), a few hours |
| Good for | repeated runs | one overnight run | one overnight run | no local GPU |

On Apple Silicon, microWakeWord turns off the Metal GPU on purpose (it is slower than the
CPU for this model), so a Mac trains on CPU like any machine without NVIDIA. Training is
20,000 steps. To estimate the total time, time the first 500-step evaluation in the log and
multiply by 40. If that comes out at more than a night, use Colab.

## Requirements

- **Python 3.12** for training. TensorFlow has no wheels for 3.13+, and microWakeWord needs
  3.10+. Sample generation alone works on 3.9-3.12.
- **ffmpeg** (recommended): resamples TTS output to 16 kHz. Without it the scripts fall back
  to scipy.
- **git**, plus **Rust/cargo** to verify and install the model with havoice.
- **Disk:** about **30 GB** free for `training_run/`. Most of it is the negative datasets
  (about 6 GB of zips, extracted to more than 20 GB).
- **RAM:** 16 GB or more is comfortable.
- **Network:** the first run downloads about 6.5 GB from Hugging Face.

### Ubuntu

Ubuntu 24.04 ships Python 3.12:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip ffmpeg git build-essential
```

On Ubuntu 22.04, get 3.12 from the deadsnakes PPA first:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt install -y python3.12 python3.12-venv
```

For a GPU you need only a working NVIDIA driver (`nvidia-smi` lists the card). The CUDA and
cuDNN libraries come from pip (see `tensorflow[and-cuda]` below), so don't install a system
CUDA toolkit for this.

### macOS

```bash
brew install python@3.12 ffmpeg git
```

The python.org 3.12 installer works too. Use `python3.12` explicitly: Homebrew's plain
`python3` may be 3.13 or newer.

Optional: the Czech system voice for `--say`. Open *System Settings → Accessibility → Spoken
Content → System voice → Manage Voices…*, add **Zuzana** under Czech, then check it with
`say -v Zuzana "Ahoj Hugo"`.

## 1. Set up the environment

A single venv covers both sample generation and training:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install piper-tts edge-tts 'datasets<4' tensorboard
git clone https://github.com/kahrendt/microWakeWord
python3 -m pip install -e ./microWakeWord
```

- `datasets<4`: version 4 changed audio decoding and breaks the download steps. The script
  refuses to start with it installed.
- `tensorboard`: microWakeWord uses `tf.summary` but doesn't declare the dependency. Without
  it, training crashes at the first evaluation with
  `TBNotInstalledError: TensorBoard is not installed`, after all the downloads and feature
  generation have already run.
- `./microWakeWord` is ignored by git. `git -C microWakeWord pull` updates it.

**Ubuntu with an NVIDIA GPU:** add the CUDA build of TensorFlow and check that it sees the
card:

```bash
python3 -m pip install 'tensorflow[and-cuda]'
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

An empty list means training will run on the CPU. That usually comes down to the driver: it
must be recent enough for the CUDA version TensorFlow ships with.

Check the whole stack before committing hours to it:

```bash
python3 -c "import microwakeword, tensorflow as tf, tensorboard, datasets, piper; print(tf.__version__, datasets.__version__)"
```

## 2. Generate positive samples

```bash
python3 training/generate_samples.py --count 1000 --edge 300 --out samples/ahoj_hugo
```

This synthesizes "Ahoj Hugo" (and variants: "Ahoj, Hugo.", "No ahoj, Hugo.", …) with the
two Czech Piper voices (jirka, kasandra), randomizing pace and noise. Voices download to
`samples/voices/` on the first run. The output is 16 kHz mono WAVs in `samples/ahoj_hugo/`
plus `samples/ahoj_hugo.zip`.

| option | adds | notes |
|---|---|---|
| `--count N` | N Piper samples (default 1000) | CPU only, a few minutes |
| `--edge N` | N Microsoft edge-tts samples (Antonin/Vlasta neural voices) | needs internet; a second engine generalizes much better |
| `--say N` | N macOS `say -v Zuzana` samples | macOS only, needs the Zuzana voice |
| `--extra-wavs DIR` | your own recordings (any rate, resampled) | **worth far more than synthetic ones** |
| `--seed N` | a different random variation | |

Real recordings make the biggest difference. Record 20-50 takes of "Ahoj Hugo" from
different distances, speeds and people, as WAVs in one folder, and pass
`--extra-wavs that/folder`. On macOS, QuickTime or Voice Memos (exported to WAV with
ffmpeg) is enough. On Ubuntu, use
`arecord -f S16_LE -r 16000 -c 1 take01.wav` (from `alsa-utils`).

Re-running adds files to the same folder (they are numbered per source and overwritten on
the next run) and rebuilds the zip. To start clean, delete `samples/ahoj_hugo*`.

piper-sample-generator, which the upstream microWakeWord notebook uses, is not used here: it
has no Czech voice checkpoints.

## 3. Train

```bash
export HF_TOKEN=hf_...      # optional: avoids Hugging Face rate limits, may unlock AudioSet
python3 training/train_ahoj_hugo.py --samples samples/ahoj_hugo.zip
```

Long runs: use `tmux`/`screen` on Ubuntu, and on macOS prefix the command with
`caffeinate -i` so the Mac doesn't sleep.

The script works inside `training_run/` (`--workdir`) and writes the result to
`ahoj_hugo_model/` (`--out`). It runs these steps:

| step | creates in `training_run/` | size |
|---|---|---|
| unpack your samples | `generated_samples/` | ~50 MB |
| room impulse responses (MIT) | `mit_rirs/` | ~6 MB |
| background noise: AudioSet shard (skipped if unavailable) + FMA music | `audioset_16k/`, `fma/`, `fma_16k/` | ~0.5 GB |
| negative speech and noise features (pre-computed by microWakeWord) | `negative_datasets/` | 25+ GB |
| augment the samples and compute spectrograms | `generated_augmented_features/` | small |
| train `mixednet` (15k steps at lr 1e-3, then 5k at 1e-4), quantize to int8 | `trained_models/wakeword/` | small |
| copy the quantized `.tflite`, write the manifest | `../ahoj_hugo_model/` | ~60 KB |

Follow the training curves in a second terminal:

```bash
.venv/bin/tensorboard --logdir training_run/trained_models/wakeword/logs
```

### Re-running and resuming

- Downloads are skipped once their target folder has content, so a second run starts almost
  straight away at feature generation. `--skip-download-data` skips the download stage
  completely.
- Training restores the newest checkpoint from `trained_models/wakeword/restore/`. To train
  from scratch after changing your samples, delete `training_run/generated_samples` (the
  samples are only unpacked when it is empty) and `training_run/trained_models`.
- **An interrupted download is not repaired.** Files are only checked for existence, and
  `negative_datasets/` is skipped as soon as the folder exists. If the first run stopped
  while downloading or unzipping, delete the incomplete pieces before running again:
  ```bash
  rm -rf training_run/negative_datasets   # if a zip is missing or its folder is absent
  rm -rf training_run/fma training_run/fma_16k
  ```
  `ls training_run/negative_datasets` should show four zips and four folders:
  `dinner_party`, `dinner_party_eval`, `no_speech`, `speech`.
- An empty `audioset_16k/` means AudioSet couldn't be fetched. Training continues with FMA
  only; setting `HF_TOKEN` may help on the next run.

## 4. Verify and flash with havoice

Flashing needs ESPHome. The Voice PE firmware pins its external components to an ESPHome dev
commit, so a release is too old; install the dev branch into its own venv, which havoice
finds on its own:

```bash
python3.12 -m venv .venv-esphome
.venv-esphome/bin/pip install "https://github.com/esphome/esphome/archive/dev.zip"
```

Then:

```bash
cargo run --example wake_word -- verify ahoj_hugo_model/ahoj_hugo.json
cargo run --example wake_word -- install ahoj_hugo_model/ahoj_hugo.json --flash
cargo run --example wake_word -- listen
```

`verify` checks the manifest and the model (int8 `[1, stride, 40]` input, uint8 output,
supported ops, arena size) without changing anything. `install --flash` stores it in
`firmware/models/`, writes the firmware YAML, builds, and uploads over Wi-Fi. The first build downloads about 1 GB of
toolchain and takes 10-20 minutes. `listen` prints every detection, so you can test the
model by saying the phrase.

## Tuning

Expect a few rounds:

| symptom | fix |
|---|---|
| misses wake-ups | lower the cutoff: `install … --cutoff 0.9` (the default in the manifest is 0.97) |
| false triggers | raise the cutoff; retrain with more varied positives so the model needs less leeway |
| works for TTS, not for you | add real recordings with `--extra-wavs` and retrain |
| works up close, not across the room | record takes from a distance; the RIR augmentation helps only so much |

Changing the cutoff needs no retraining: re-run `install` with `--cutoff` and flash again.

## License note

The augmentation and negative datasets (AudioSet, FMA, microWakeWord's negatives) mix
licenses. Treat the trained model as for non-commercial, personal use.
