# Training an "Ahoj Hugo" wake word model

The Voice PE only runs [microWakeWord](https://github.com/kahrendt/microWakeWord) models
(int8 streaming TFLite, 40-bin spectrogram input). An openWakeWord model can't be converted
into one; it has to be trained from audio. The Hugo assistant runs an
[openWakeWord](https://github.com/dscripka/openWakeWord) ONNX classifier instead. Training
builds both from the same samples. This directory covers every step, from no data to a
flashed device:

```
generate_samples.py ──► samples/ahoj_hugo.zip ──► train_ahoj_hugo.py ──► ahoj_hugo_model/
   (Piper, edge-tts,        1000+ 16 kHz WAVs       (or the Colab          ahoj_hugo.tflite ─► Voice PE
    say, your voice)                                 notebook:             ahoj_hugo.json
                                                     .tflite only)         ahoj_hugo.onnx ──► Hugo assistant
                                                                                 │
                                         cargo run --example wake_word -- install --flash
```

| file | purpose |
|---|---|
| `generate_samples.py` | Czech samples: "Ahoj Hugo" + "Ahoj Google" positives, optional negatives (Piper voices, edge-tts, macOS `say`, real recordings) |
| `train_ahoj_hugo.py` | local training of both models: downloads, augmentation, features, training, quantization, manifest |
| `train_oww.py` | the ONNX step on its own (run by `train_ahoj_hugo.py`) |
| `mww_train.py` | runs microWakeWord's training with small-GPU fixes (run by `train_ahoj_hugo.py`) |
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

- **Python 3.12 or 3.14.** microWakeWord needs 3.10+. Python 3.14 is tested on Ubuntu 26.04
  with an NVIDIA GPU. There TensorFlow comes from the 2.22 release candidate, the first build
  with 3.14 wheels. Sample generation works on 3.9 and newer.
- **ffmpeg** (recommended): resamples TTS output to 16 kHz. Without it the scripts fall back
  to scipy.
- **git**, plus **Rust/cargo** to verify and install the model with havoice.
- **Disk:** about **30 GB** free for `training_run/`. Most of it is the negative datasets
  (about 6 GB of zips, extracted to more than 20 GB).
- **RAM:** 16 GB or more is comfortable.
- **Network:** the first run downloads about 6.5 GB from Hugging Face.

### Ubuntu

Ubuntu 26.04 ships Python 3.14:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg git build-essential
```

Ubuntu 24.04 ships Python 3.12:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip ffmpeg git build-essential
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

A single venv covers both sample generation and training. On Ubuntu 26.04 create it with
`python3 -m venv .venv` (Python 3.14). Elsewhere use 3.12:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install piper-tts edge-tts 'datasets<4' tensorboard
git clone https://github.com/kahrendt/microWakeWord
python3 -m pip install -e ./microWakeWord
```

- `datasets<4`: version 4 changed audio decoding and breaks the download steps. The script
  refuses to start with it installed. On Python 3.14, `datasets` 3.x hashing crashes with
  `TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given`.
  `train_ahoj_hugo.py` patches this at startup, so keep the pinned `dill`/`multiprocess`
  versions that `datasets` installs. Upgrading them doesn't help.
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

On Python 3.14, pip picks `tensorflow` 2.22.0rc0 by itself: no stable release has 3.14
wheels yet, and `--pre` isn't needed. On Ubuntu 26.04 it finds the GPU with the cuDNN that
`[and-cuda]` installs.

An empty list means training will run on the CPU. That usually comes down to the driver: it
must be recent enough for the CUDA version TensorFlow ships with.

A 4 GB GPU (for example a laptop RTX 500) is enough. microWakeWord's validation passes the
whole ambient validation set to `model.evaluate()` at once, and Keras copies it to the GPU in
one piece. On a small card that fails at the first evaluation (step 500) with
`InternalError: Failed copying input tensor from CPU:0 to GPU:0 ... Dst tensor is not
initialized`. `train_ahoj_hugo.py` therefore runs training through `training/mww_train.py`,
which feeds those arrays to the GPU batch by batch and turns on GPU memory growth. The
metrics are the same as Keras' own `evaluate()`, and microWakeWord stays unmodified.

Validation is also the step that needs the most RAM: the ambient set is about 3.8 GB as
float32. With `mww_train.py` the training process stays at about 6.5 GB for the whole run.
An earlier version of the launcher used `tf.data.Dataset.from_generator`, which kept every
validation set alive. It added 3.8 GB per evaluation, so the kernel killed training
(`died with <Signals.SIGKILL: 9>`) at about step 8500 on a 64 GB machine.

Check the whole stack before committing hours to it:

```bash
python3 -c "import microwakeword, tensorflow as tf, tensorboard, datasets, piper; print(tf.__version__, datasets.__version__)"
```

## 2. Generate samples

```bash
python3 training/generate_samples.py --count 1000 --multi 2000 --edge 300 --out samples/ahoj_hugo
```

This synthesizes both wake phrases, split evenly: "Ahoj Hugo" and "Ahoj Google", each with
variants ("Ahoj, Hugo.", "No ahoj, Google.", …). Speed, noise and pitch are randomized.
Voices download to `samples/voices/` on the first run (about 700 MB for all of them). The
output is 16 kHz mono WAVs in `samples/ahoj_hugo/` plus `samples/ahoj_hugo.zip`.

The more different voices, the better the model copes with real people:

- **`--count`** uses eight single-speaker Piper voices: Czech jirka (medium and low) and
  kasandra, Slovak lili, and Polish darkman, gosia, mc_speech and bass. All of them get
  Czech pronunciation. The Slovak and Polish voices add new timbres, and the Czech ones are
  picked most often (7 in 12 samples).
- **`--multi`** uses multi-speaker Piper models with Czech pronunciation: US English
  LibriTTS-R (904 speakers) and British VCTK (109 speakers). Every sample gets a random
  speaker, so 2000 samples cover about 900 different voices, all with an English accent.
  German MLS (236 speakers) was tried and dropped: with Czech phonemes it babbles.
- **Pitch shifting:** 30% of the synthesized samples are shifted up to 4 semitones up or
  down (`--pitch-prob`, `--pitch-range`). Pitch and formants move together and the length
  stays the same, so it sounds like a different, deeper or higher voice. This needs
  ffmpeg.
- **Spelling of "Google":** Czech text-to-speech reads "Google" letter by letter
  ("go-og-le"), so Piper and `say` get the Czech spelling "Gůgl". edge-tts gets both
  spellings.

| option | adds | notes |
|---|---|---|
| `--count N` | N samples from the 8 single-speaker Piper voices (default 1000) | CPU only, runs on all cores |
| `--multi N` | N samples from 1013 multi-speaker Piper voices (default 0) | the cheapest way to add many voices |
| `--edge N` | N Microsoft edge-tts samples (Antonin/Vlasta neural voices) | needs internet; a second engine generalizes much better |
| `--say N` | N macOS `say -v Zuzana` samples | macOS only, needs the Zuzana voice |
| `--wake-words hugo google` | which phrases to generate (default both) | `--wake-words hugo` for "Ahoj Hugo" only |
| `--pitch-prob P` / `--pitch-range S` | pitch/formant-shift a share P of samples by up to ±S semitones (default 0.3 / 4) | `--pitch-prob 0` turns it off |
| `--negatives N --negative-phrase "…"` | N samples of phrases that must **not** wake the device | see below |
| `--extra-wavs DIR` | your own recordings (any rate, resampled) | **worth far more than synthetic ones** |
| `--seed N` | a different random variation | |
| `--jobs N` | N parallel Piper worker processes (default: all CPU cores) | about 4× faster on 16 cores than one process |
| `--edge-jobs N` | N parallel edge-tts requests (default 4) | raise carefully: Microsoft throttles bursts |

**Negative samples** cut false wake-ups on phrases that sound close to the wake words.
Name the phrases (repeat the option for several) and how many samples to make. They are
split across Piper, multi-speaker and edge-tts in the same ratio as the positives, and are
written to `samples/negatives/` (`--negatives-out`) plus a zip:

```bash
python3 training/generate_samples.py --count 1000 --multi 2000 --edge 300 \
    --negatives 1000 --negative-phrase "Ahoj kluci." --negative-phrase "Ahoj Hubo."
```

Pass the zip to training with `--negatives samples/negatives.zip`. Don't use a phrase you
want to wake on: "Ahoj Google" is a positive phrase here.

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
# with negative samples from step 2:
python3 training/train_ahoj_hugo.py --samples samples/ahoj_hugo.zip --negatives samples/negatives.zip
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
| unpack negative samples (only with `--negatives`) | `generated_negatives/` | small |
| augment the samples and compute spectrograms | `generated_augmented_features/` | small |
| same for the negative samples (only with `--negatives`) | `generated_negative_features/` | small |
| train `mixednet` (15k steps at lr 1e-3, then 5k at 1e-4), quantize to int8 | `trained_models/wakeword/` | small |
| copy the quantized `.tflite`, write the manifest | `../ahoj_hugo_model/` | ~60 KB |

Follow the training curves in a second terminal:

```bash
.venv/bin/tensorboard --logdir training_run/trained_models/wakeword/logs
```

### Re-running and resuming

- The four download steps (RIRs, AudioSet, FMA, negative datasets) run at the same time,
  and the four negative zips download in parallel, so their progress lines interleave.
- Downloads are skipped once their target folder has content, so a second run starts almost
  straight away at feature generation. `--skip-download-data` skips the download stage
  completely.
- Training restores the newest checkpoint from `trained_models/wakeword/restore/`. A
  checkpoint is written only when validation improves, so a crashed run resumes from its
  best weights, not from the step where it stopped. The step counter isn't saved either:
  the resumed run goes through all 20,000 steps again, starting from those weights. To train
  from scratch after changing your samples, delete `training_run/generated_samples` and
  `training_run/generated_negatives` (samples are only unpacked into an empty folder) and
  `training_run/trained_models`.
- **An interrupted download restarts; an interrupted unzip or decode does not.** Files
  download to `*.part` and are renamed only when complete, so a cut-off zip is fetched again
  on the next run. Each negative dataset is skipped once its folder exists, though, and
  `fma_16k/`, `audioset_16k/` and `mit_rirs/` once they hold any WAV. If the first run
  stopped while unzipping or converting, delete the incomplete pieces before running again:
  ```bash
  rm -rf training_run/negative_datasets/speech   # any half-extracted folder
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
.venv/bin/pip install "https://github.com/esphome/esphome/archive/dev.zip"
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
| false triggers on one similar phrase | add it as `--negatives N --negative-phrase "…"` and retrain with `--negatives` |
| works for some voices only | add `--multi 2000` (or more) for many more speakers |
| works for TTS, not for you | add real recordings with `--extra-wavs` and retrain |
| works up close, not across the room | record takes from a distance; the RIR augmentation helps only so much |

Changing the cutoff needs no retraining: re-run `install` with `--cutoff` and flash again.

## License note

The augmentation and negative datasets (AudioSet, FMA, microWakeWord's negatives) mix
licenses. Treat the trained model as for non-commercial, personal use.
