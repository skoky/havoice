//! Library for playing with the Home Assistant Voice Preview Edition.
//!
//! * [`VoicePe`] - high-level handle: LED ring, audio, speech, settings, typed events.
//! * [`api::Client`] - lower-level ESPHome native API client (built on `esphome-client`).
//! * [`audio::Pcm`] - generate / load 16-bit PCM, encoded to FLAC for the device.
//! * [`improv`] - Wi-Fi setup over USB (Improv Serial), [`VoicePe::pair`] - set the API key.
//! * [`discover`] - mDNS discovery, [`serial`] - USB log tail, [`serve`] - local file server.

pub mod api;
pub mod audio;
pub mod device;
pub mod discover;
pub mod improv;
pub mod serial;
pub mod serve;
pub mod tts;

pub use audio::Pcm;
pub use device::{Event, Events, MediaPlayerCommand, Rgb, VoicePe};
/// Raw ESPHome protobuf messages, for anything the high-level API doesn't cover.
pub use esphome_client::types as proto;
