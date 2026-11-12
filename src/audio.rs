//! PCM helpers. The Voice PE firmware decodes FLAC and MP3 but *not* WAV, so anything
//! generated on this computer is encoded to FLAC (pure Rust, via `flacenc`) before it's
//! served to the device.
//!
//! ```no_run
//! # async fn demo() -> anyhow::Result<()> {
//! use havoice::audio::{Pcm, envelope, sine};
//!
//! // a one-second 440 Hz beep with a soft attack and release
//! let mut pcm = Pcm::new(22_050);
//! pcm.push(1.0, |t| 0.5 * sine(440.0, t) * envelope(t, 1.0, 0.02, 0.3));
//! havoice::VoicePe::from_env().await?.play_pcm(&pcm, true).await?;
//! # Ok(()) }
//! ```

use anyhow::{Context, Result, anyhow, bail, ensure};
use flacenc::component::BitRepr;
use flacenc::error::Verify;

/// Mono 16-bit PCM.
#[derive(Debug, Clone, Default)]
pub struct Pcm {
    /// Sample rate in Hz.
    pub rate: u32,
    /// Signed 16-bit samples.
    pub samples: Vec<i16>,
}

impl Pcm {
    /// Empty audio at `rate` Hz (22 050 is plenty for speech and beeps).
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            samples: Vec::new(),
        }
    }

    /// Duration in seconds.
    pub fn seconds(&self) -> f32 {
        self.samples.len() as f32 / self.rate as f32
    }

    /// Append `f(t)` for `secs` seconds; `f` returns -1.0..=1.0 and gets time since the
    /// start of this segment.
    pub fn push(&mut self, secs: f32, mut f: impl FnMut(f32) -> f32) -> &mut Self {
        let n = (secs * self.rate as f32) as usize;
        let dt = 1.0 / self.rate as f32;
        self.samples.extend((0..n).map(|i| {
            let v = f(i as f32 * dt).clamp(-1.0, 1.0);
            (v * i16::MAX as f32) as i16
        }));
        self
    }

    /// Append `secs` seconds of silence.
    pub fn silence(&mut self, secs: f32) -> &mut Self {
        self.push(secs, |_| 0.0)
    }

    /// Encode as a FLAC file.
    pub fn to_flac(&self) -> Result<Vec<u8>> {
        ensure!(!self.samples.is_empty(), "no audio to encode");
        let samples: Vec<i32> = self.samples.iter().map(|&s| s as i32).collect();
        let config = flacenc::config::Encoder::default()
            .into_verified()
            .map_err(|(_, e)| anyhow!("flac config: {e:?}"))?;
        let source = flacenc::source::MemSource::from_samples(&samples, 1, 16, self.rate as usize);
        let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
            .map_err(|e| anyhow!("flac encode: {e:?}"))?;
        let mut sink = flacenc::bitsink::ByteSink::new();
        stream
            .write(&mut sink)
            .map_err(|e| anyhow!("flac write: {e:?}"))?;
        Ok(sink.as_slice().to_vec())
    }

    /// Parse a 16-bit PCM WAV (mono, or the left channel of stereo).
    pub fn from_wav(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
            "not a WAV file"
        );
        let (mut rate, mut channels, mut bits) = (0u32, 0u16, 0u16);
        let mut pos = 12;
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
            let body = bytes
                .get(pos + 8..(pos + 8 + len).min(bytes.len()))
                .context("truncated WAV")?;
            match id {
                b"fmt " => {
                    ensure!(body.len() >= 16, "short fmt chunk");
                    channels = u16::from_le_bytes([body[2], body[3]]);
                    rate = u32::from_le_bytes(body[4..8].try_into()?);
                    bits = u16::from_le_bytes([body[14], body[15]]);
                }
                b"data" => {
                    ensure!(
                        bits == 16 && channels > 0,
                        "need 16-bit PCM, got {bits}-bit"
                    );
                    let step = 2 * channels as usize;
                    let samples = body
                        .chunks_exact(step)
                        .map(|f| i16::from_le_bytes([f[0], f[1]]))
                        .collect();
                    return Ok(Self { rate, samples });
                }
                _ => {}
            }
            pos += 8 + len + (len & 1);
        }
        bail!("WAV has no data chunk")
    }
}

/// Sine wave.
pub fn sine(freq: f32, t: f32) -> f32 {
    (std::f32::consts::TAU * freq * t).sin()
}

/// Simple attack/decay envelope over a note of length `len`.
pub fn envelope(t: f32, len: f32, attack: f32, release: f32) -> f32 {
    let a = (t / attack).min(1.0);
    let r = ((len - t) / release).clamp(0.0, 1.0);
    a * r
}

/// Equal-tempered frequency of a MIDI note (69 = A4 = 440 Hz).
pub fn midi(note: u8) -> f32 {
    440.0 * 2f32.powf((note as f32 - 69.0) / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip_and_flac_header() {
        let mut pcm = Pcm::new(8000);
        pcm.push(0.1, |t| sine(440.0, t));
        let data: Vec<u8> = pcm.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut wav = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        wav.extend(16u32.to_le_bytes());
        wav.extend([1, 0, 1, 0]); // PCM, mono
        wav.extend(8000u32.to_le_bytes());
        wav.extend(16000u32.to_le_bytes());
        wav.extend([2, 0, 16, 0]);
        wav.extend(b"data");
        wav.extend((data.len() as u32).to_le_bytes());
        wav.extend(&data);

        let back = Pcm::from_wav(&wav).unwrap();
        assert_eq!(back.rate, 8000);
        assert_eq!(back.samples, pcm.samples);
        assert_eq!(&back.to_flac().unwrap()[..4], b"fLaC");
    }
}
