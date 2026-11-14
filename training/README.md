# Training an "Ahoj Hugo" wake word model

The Voice PE only runs microWakeWord models (int8 TFLite, 40-bin melspectrogram input), and
no amount of converting gets an openWakeWord model there - it has to be trained from audio.
This directory is the full path from zero data to a flashed device:

1. **Generate samples on your Mac** (no GPU needed):

   ```sh
   python3 -m pip install piper-tts
   python3 training/generate_samples.py --count 1000 --out samples/ahoj_hugo
   ```

   This synthesizes "Ahoj Hugo" with the two Czech Piper voices (jirka, kasandra) with
   randomized pitch/pace/noise, resamples to 16 kHz, and zips the result. Strongly
   recommended additions:

   - `--edge 300` - 300 more samples from Microsoft edge-tts (cs-CZ Antonin/Vlasta neural
     voices; a second TTS engine generalizes better). `python3 -m pip install edge-tts`.
   - `--say 100` - the macOS Czech voice (`say -v Zuzana`).
   - `--extra-wavs ../oww_rs/crates/oww/test_data` - folds in real recordings of you
     saying the phrase (ahoj_hugo_man.wav, ahoj_hugo_woman.wav).

   piper-sample-generator is *not* used: it only ships voice checkpoints for English,
   German, French and Dutch, none for Czech.

2. **Train on Google Colab**: open `training/train_ahoj_hugo.ipynb`, switch the runtime to
   GPU, upload the zip, and run the cells top to bottom (a few hours). The notebook wraps
   [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord): it augments the
   samples with room impulse responses and background noise, mixes in pre-generated
   negative speech/noise features, trains a streaming `mixednet`, quantizes it, and hands
   you `ahoj_hugo_model.zip` with `ahoj_hugo.tflite` + `ahoj_hugo.json`.

3. **Verify and flash with havoice**:

   ```sh
   cargo run --example wake_word -- verify ahoj_hugo.tflite --wake-word "Ahoj Hugo"
   cargo run --example wake_word -- install ahoj_hugo.tflite --wake-word "Ahoj Hugo" --flash
   ```

Expect to iterate. If it misses wake-ups, lower `probability_cutoff` in the manifest; if it
false-triggers, raise it, add more negative Czech speech, or retrain with more/varied
positive samples (`--edge`, `--say`, your own recordings). The augmentation datasets
(AudioSet, FMA) mix licenses, so treat the trained model as non-commercial, personal use.

Files:

| file | purpose |
|---|---|
| `generate_samples.py` | Mac-side Czech sample generation (Piper voices, edge-tts, `say`, real recordings) |
| `train_ahoj_hugo.ipynb` | Colab notebook: augmentation, training, quantization, manifest, download |
