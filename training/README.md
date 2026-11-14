# Training an "Ahoj Hugo" wake word model

The Voice PE only runs microWakeWord models (int8 TFLite, 40-bin melspectrogram input), and
no amount of converting gets an openWakeWord model there - it has to be trained from audio.
This directory is the full path from zero data to a flashed device:

1. **Generate samples on your computer** (Linux or macOS, no GPU needed, Python 3.9-3.12):

   ```sh
   python3 -m pip install piper-tts
   python3 training/generate_samples.py --count 1000 --out samples/ahoj_hugo
   ```

   This synthesizes "Ahoj Hugo" with the two Czech Piper voices (jirka, kasandra) with
   randomized pitch/pace/noise, resamples to 16 kHz (ffmpeg or scipy), and zips the result.
   Strongly recommended additions:

   - `--edge 300` - 300 more samples from Microsoft edge-tts (cs-CZ Antonin/Vlasta neural
     voices; a second TTS engine generalizes better). `python3 -m pip install edge-tts`.
   - `--say 100` - the macOS Czech voice (`say -v Zuzana`), macOS only.
   - `--extra-wavs DIR` - folds in real recordings of you saying the phrase; real samples
     are worth far more than synthetic ones.

   piper-sample-generator is *not* used: it only ships voice checkpoints for English,
   German, French and Dutch, none for Czech.

2. **Train** - either way runs [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord):
   it augments the samples with room impulse responses and background noise, mixes in
   pre-generated negative speech/noise features, trains a streaming `mixednet`, quantizes
   it, and produces `ahoj_hugo.tflite` + `ahoj_hugo.json`.

   - **Google Colab** (recommended): open `training/train_ahoj_hugo.ipynb`, switch the
     runtime to GPU, upload the zip, run the cells top to bottom - a few hours.
   - **Locally** with `training/train_ahoj_hugo.py` - practical only with an NVIDIA GPU
     (check `nvidia-smi`); on CPU expect days. The training stack needs **Python 3.12**
     (tensorflow has no release for newer versions), and `tensorboard` which
     microwakeword forgets to declare:

     ```sh
     python3.12 -m venv .venv
     . .venv/bin/activate
     python3 -m pip install 'datasets<4' tensorboard
     git clone https://github.com/kahrendt/microWakeWord
     python3 -m pip install -e ./microWakeWord
     export HF_TOKEN=hf_...            # optional, avoids HF rate limits
     python3 training/train_ahoj_hugo.py --samples samples/ahoj_hugo.zip
     ```

     Re-running is safe: downloads and features are reused, and training resumes from the
     last checkpoint.

3. **Verify and flash with havoice** (ESPHome required for `--flash`; the Voice PE firmware
   pins external components to an ESPHome dev commit, so install the dev branch, a release
   is too old):

   ```sh
   python3.12 -m venv .venv-esphome
   .venv-esphome/bin/pip install "https://github.com/esphome/esphome/archive/dev.zip"
   cargo run --example wake_word -- verify ahoj_hugo_model/ahoj_hugo.json
   cargo run --example wake_word -- install ahoj_hugo_model/ahoj_hugo.json --flash
   ```

   havoice finds `.venv-esphome/bin/esphome` on its own. The first build downloads ~1 GB
   of toolchain (10-20 minutes) and uploads over Wi-Fi.

Expect to iterate. If it misses wake-ups, lower `probability_cutoff` (`--cutoff 0.9` on
install); if it false-triggers, raise it, add more negative Czech speech, or retrain with
more/varied positive samples (`--edge`, `--say`, your own recordings). The augmentation
datasets (AudioSet, FMA) mix licenses, so treat the trained model as non-commercial,
personal use.

Files:

| file | purpose |
|---|---|
| `generate_samples.py` | Czech sample generation (Piper voices, edge-tts, `say`, real recordings) |
| `train_ahoj_hugo.ipynb` | Colab notebook: augmentation, training, quantization, manifest, download |
| `train_ahoj_hugo.py` | the same pipeline as a local script (needs a GPU to be practical) |
