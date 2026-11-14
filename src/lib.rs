//! Drive a **Home Assistant Voice Preview Edition** (Voice PE) from Rust over its ESPHome
//! native API - next to Home Assistant, or without it.
//!
//! ```no_run
//! use havoice::{Event, Rgb, VoicePe};
//!
//! # async fn demo() -> anyhow::Result<()> {
//! // HAVOICE_HOST (or mDNS discovery) + HAVOICE_KEY (the API encryption key), also from .env
//! let pe = VoicePe::from_env().await?;
//!
//! pe.led(Rgb(255, 100, 0), 0.8).await?;                  // LED ring
//! pe.say("The laundry is done").await?;                   // speech (macOS say / espeak-ng)
//! pe.play_url("https://example.com/radio.mp3", false).await?; // MP3 / FLAC streams
//! pe.set_volume(0.4).await?;
//!
//! let mut events = pe.events().await?;                    // buttons, dial, mute, media ...
//! while let Some(ev) = events.next().await {
//!     if let Event::Button(press) = ev {
//!         println!("{press}");                            // double_press, triple_press, long_press
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! # What's in here
//!
//! * [`VoicePe`] - the high-level handle: LED ring, speaker (URLs, local files, generated
//!   audio, speech), volume, entity settings, typed [`Event`]s, and [`VoicePe::pair`] to give a
//!   factory-reset device its API key without Home Assistant.
//! * [`audio::Pcm`] - generate or load 16-bit PCM; it's sent to the device as FLAC (the stock
//!   firmware plays FLAC and MP3, not WAV).
//! * [`tts`] - text-to-speech with a choice of voice.
//! * [`wake_word`] - check microWakeWord models, build them into firmware, watch detections.
//! * [`onnx`] - read an `.onnx` model and judge how far it is from a microWakeWord `.tflite`.
//! * [`improv`] - Wi-Fi setup over the USB cable (Improv Serial); [`serial`] - the USB log.
//! * [`discover`] - mDNS discovery; [`serve`] - the one-file HTTP server the device streams from.
//! * [`api::Client`] and [`proto`] - the raw ESPHome API for everything else.
//!
//! # Connecting
//!
//! The stock firmware encrypts its API with a key that Home Assistant generated when it adopted
//! the device. Copy it from Home Assistant (the repository README shows how) into
//! `HAVOICE_KEY`, or reset the device and let [`VoicePe::pair`] set a new one. Several API
//! clients may be connected at once, so this crate works alongside Home Assistant.
//!
//! # Platform notes
//!
//! Developed on macOS, also builds on Linux (`libudev-dev` for the serial port). Playing local
//! audio needs the device to reach this computer over HTTP; speech uses `say` on macOS and
//! `espeak-ng` elsewhere. The `cli` feature (on by default) builds the `havoice` command-line
//! tool; turn default features off to use only the library.

#![warn(missing_docs)]

pub mod api;
pub mod audio;
pub mod device;
pub mod discover;
pub mod improv;
pub mod onnx;
pub mod serial;
pub mod serve;
pub mod tts;
pub mod wake_word;

pub use audio::Pcm;
pub use device::{Event, Events, MediaPlayerCommand, Rgb, VoicePe};
/// Raw ESPHome protobuf messages, for anything the high-level API doesn't cover.
pub use esphome_client::types as proto;
