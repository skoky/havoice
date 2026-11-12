# havoice

[![crates.io](https://img.shields.io/crates/v/havoice.svg)](https://crates.io/crates/havoice)
[![docs.rs](https://img.shields.io/docsrs/havoice)](https://docs.rs/havoice)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Drive a **[Home Assistant Voice Preview Edition](https://www.home-assistant.io/voice-pe/)** (Voice PE)
from Rust - a library, a command-line tool and a set of examples. It talks to the device's ESPHome native API
directly, so it works **next to Home Assistant or without it**.

* LED ring: colours, fades, the firmware's effects
* Speaker: internet radio and other MP3/FLAC URLs, local files, audio generated in Rust, speech in a voice you pick
* Volume, mute, and every entity the firmware exposes (selects, switches, numbers, buttons)
* Events: button presses, the dial, the mute switch, media player state
* Wake words: list the models on the device, watch it detect them, check and build new
  [microWakeWord](https://github.com/kahrendt/microWakeWord) models into its firmware
* Setup without Home Assistant: Wi-Fi over the USB cable (Improv Serial) and a fresh API key

```rust
use havoice::{Event, Rgb, VoicePe};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pe = VoicePe::from_env().await?;   // HAVOICE_HOST (or mDNS) + HAVOICE_KEY, also read from .env

    pe.led(Rgb(255, 100, 0), 0.8).await?;
    pe.say("The laundry is done").await?;  // macOS `say` / Linux `espeak-ng`, sent to the device as FLAC
    pe.set_volume(0.4).await?;

    let mut events = pe.events().await?;
    while let Some(ev) = events.next().await {
        if let Event::Button(press) = ev {
            println!("{press}");           // double_press, triple_press, long_press
        }
    }
    Ok(())
}
```

## Install

```toml
[dependencies]
havoice = { version = "0.1", default-features = false }   # the library only
tokio = { version = "1", features = ["full"] }
anyhow = "1"
```

The `havoice` command-line tool (the default `cli` feature):

```bash
cargo install havoice
```

To run the examples, clone the repository and copy `.env.example` to `.env`.

**Linux** needs `libudev` for the serial port and, for speech, `espeak-ng`:

```bash
sudo apt install build-essential pkg-config libudev-dev cmake espeak-ng
```

## Connecting

Two settings, from the environment or a `.env` file (variables already set in the shell win):

| variable | |
|---|---|
| `HAVOICE_HOST` | IP or `.local` name of the device; empty = the first Voice PE found via mDNS |
| `HAVOICE_KEY` | the API encryption key (base64) |

The stock firmware encrypts its API with a key Home Assistant generated when it adopted the device. It is
**not** your Home Assistant password. To get it:

* **HA OS / Supervised** - in the *Terminal & SSH* add-on:
  ```bash
  jq -r '.data.entries[] | select(.domain=="esphome") | "\(.title)  \(.data.host)  \(.data.noise_psk)"' /config/.storage/core.config_entries
  ```
* **HA Container** - the same file in your config directory, or run the command inside the container
  (`docker exec -it homeassistant ...` / `podman exec -it homeassistant ...`).
* **ESPHome Device Builder** - after *Take control*, the device YAML has `api: encryption: key:`.
* **No Home Assistant** - factory-reset the device (hold the centre button ~22 s until the red circle
  completes) and let `cargo run --example pair -- --wifi "YourSSID"` give it Wi-Fi over USB and a new key.

A wrong key fails fast with a handshake error. ESPHome accepts several API clients at once, so Home Assistant
keeps working while havoice is connected.

## Examples

| example | shows |
|---|---|
| `cargo run --example tour` | guided tour: info, entities, LED colours/fades/effects, speech, button/dial/mute events |
| `cargo run --example discover` | mDNS discovery of ESPHome devices (no key needed) |
| `cargo run --example led_show [-- ff6600\|Rainbow\|off]` | colour wheel animation, one colour, or a firmware effect |
| `cargo run --example speak -- "text"` / `-- --file x.mp3` | speech or a local file, as an announcement |
| `cargo run --example synth -- [chime\|arp\|alarm\|sweep\|waves]` | sounds generated in Rust with `havoice::Pcm` |
| `cargo run --example radio -- [--volume 30] [--voice Daniel\|cs] [jazz]` | internet radio: presets or a [radio-browser.info](https://www.radio-browser.info) search, now-playing titles, spoken station names; `+` / `-` / `v 40` for volume, `n` / `p` / a number for stations, `q` to quit; double press = next, triple = previous, long press = stop |
| `cargo run --example button_actions` | desk control pad: double press says the time, triple the battery level, long toggles the LED |
| `cargo run --example pair -- --status` | read-only: Wi-Fi state over USB and whether the device has a key |
| `cargo run --example pair [-- --wifi SSID]` | set up a factory-reset device without Home Assistant (not yet tried on real hardware) |
| `cargo run --example wake_word -- list` | wake word models on the device, stored locally, and in ESPHome's online catalogue |
| `cargo run --example wake_word -- verify <src>` | check a model the way the device does (catalogue name, URL, or local file) |
| `cargo run --example wake_word -- install <src> [--flash]` | build verified models into the firmware; `--flash` uploads it over Wi-Fi |
| `cargo run --example wake_word -- listen` | print every wake word detection with its probabilities |

## Command-line tool

| command | what |
|---|---|
| `havoice discover` | mDNS scan for ESPHome devices |
| `havoice info` / `entities` | device info / every entity with its options, effects and event types |
| `havoice watch` | live state stream: buttons, dial, mute, LEDs, media player |
| `havoice logs [--level debug]` | device log over Wi-Fi |
| `havoice serial [--port P] [--grep wake]` | device log over USB (opening the port does not reset the chip) |
| `havoice led ff6600` / `led Rainbow` / `led off` | LED ring |
| `havoice say [--voice Daniel\|cs] "hello"` / `havoice voices [cs]` | speech, and the voices installed here |
| `havoice play song.mp3 [--announce] [--volume .4]` | a URL, or a local file served from this computer |
| `havoice media pause\|stop\|volume 0.3 …` | media player |
| `havoice select …` / `switch …` / `number …` / `press restart` | any entity |

## Audio, speech and volume

The stock firmware decodes **FLAC and MP3 only** - no WAV, AAC, Ogg or HLS. Generated audio and speech are
encoded to FLAC in-process (pure Rust) and served to the device by a one-file HTTP server, so the device must be
able to reach this computer.

```rust
use havoice::{audio::{envelope, sine}, tts, Pcm};

let mut beep = Pcm::new(22_050);
beep.push(1.0, |t| 0.5 * sine(440.0, t) * envelope(t, 1.0, 0.02, 0.3));
pe.play_pcm(&beep, true).await?;                     // true = announcement, ducks music

pe.say_voice("Hello", "Daniel").await?;              // a voice by name
let czech = tts::find_voice("cs").await?;            // or the first voice of a language
pe.say_voice("Dobrý den", &czech.name).await?;

pe.set_volume(0.3).await?;
```

Speech uses `say` on macOS and `espeak-ng` elsewhere; `HAVOICE_TTS` picks another program and
`HAVOICE_TTS_VOICE` a default voice. The firmware maps volume 0-100 % onto 40-85 % of its DAC range, so 0 % is
quiet, not silent - use mute for silence. The built-in speaker is meant for voice; for music, connect a powered
speaker to the 3.5 mm jack.

## Wake word models

The stock firmware ships with *Okay Nabu*, *Hey Jarvis* and *Hey Mycroft* (plus *Stop* for timers), and
`wake_word::Detections` reports every detection from the device log.

ESPHome compiles wake word models into the firmware - it cannot receive one at runtime. So
`wake_word install` builds **custom firmware**: the stock Voice PE configuration plus your models, with the same
device name and API key, so Home Assistant and havoice keep working. Before anything is stored, each model is
checked the way the firmware checks it when loading: TFLite schema 3, only the 20 operators its interpreter
has, input int8 `[1, stride, 40]`, output uint8 `[1, 1]`, sensible manifest values, ESPHome version.

* Sources: an [ESPHome catalogue](https://github.com/esphome/micro-wake-word-models) name (`alexa`,
  `hey_home_assistant`, ...), an http(s) URL of a manifest `.json` or a `.tflite`, or a local file. A bare
  `.tflite` needs `--wake-word "..."` (optionally `--language`, `--cutoff`, `--window`, `--arena`).
* `--flash` needs the ESPHome CLI (`pip install esphome`, or set `HAVOICE_ESPHOME`). The first build downloads
  about 1 GB of toolchain and takes 10-20 minutes; later builds take a minute. The upload goes over Wi-Fi.
* Added models switch on whenever an API client connects; Home Assistant can still choose wake words as usual.
* If the device can't reach its saved Wi-Fi network, the custom firmware opens a fallback hotspot; its password is
  in `firmware/secrets.yaml`.
* Back to stock firmware: `wake_word remove <id> --flash`, or the
  [Voice PE web flasher](https://esphome.github.io/home-assistant-voice-pe/).

Only microWakeWord models run on the device. openWakeWord / ONNX models have to be retrained with microWakeWord.

## Linux and macOS

Developed and tested on macOS; it builds on Linux, where these differ:

| | macOS | Linux |
|---|---|---|
| USB port | `/dev/cu.usbmodem*` | `/dev/ttyACM*` - both found automatically by Espressif's USB vendor id; override with `--port` / `HAVOICE_SERIAL` |
| USB access | works | `sudo usermod -aG dialout $USER`, then log in again |
| Speech | `say` | `espeak-ng` |
| `.local` names | built in | Avahi (`avahi-daemon libnss-mdns` on servers); havoice falls back to its own mDNS lookup |
| Network access | System Settings → Privacy & Security → **Local Network** must allow your terminal app | if `ufw` is on, allow mDNS (`5353/udp`) and connections from the device (it fetches audio from this computer) |

On Ubuntu, ModemManager may grab a newly plugged `/dev/ttyACM0`; stop it or add a udev rule with
`ENV{ID_MM_DEVICE_IGNORE}="1"`.

## Troubleshooting

* **`No route to host` while `ping` works** - macOS is blocking your terminal app from the local network
  (`ping` and `nc` are exempt, so they prove nothing). Allow it under Privacy & Security → Local Network.
* **A `.local` name doesn't resolve** - use the IP, and give the device a fixed DHCP lease on the router.
* **Plays nothing, log shows `Failed to create socket`** - the device ran out of network sockets; restart it
  (`havoice press restart`, or unplug it).
* **No log over USB, no Improv reply, not on Wi-Fi** - the chip is probably sitting in its USB bootloader. Reset it
  from the inside: `esptool --port <port> --after watchdog-reset read-mac` (a reset via the RTS line can put it
  straight back into the bootloader).
* **`Connection reset` / timeouts** - the device sleeps its Wi-Fi between packets; havoice retries a few times.
  Repeated failures usually mean it dropped off the network (see above).

## How it works

Transport, the Noise handshake and the protobuf messages come from
[`esphome-client`](https://crates.io/crates/esphome-client) (API 1.16). The tests run havoice against the
*device* side of [`esphome-native-api`](https://crates.io/crates/esphome-native-api), so two independent
implementations have to agree; the pairing tests use a hand-written plaintext device.

```
src/device.rs      VoicePe: the high-level API and typed events
src/api.rs         connection, message fan-out, entities, error hints
src/audio.rs       PCM, FLAC encoding, WAV reading
src/tts.rs         say / espeak-ng, voices
src/wake_word.rs   model checks (a small TFLite reader), firmware YAML, detections
src/improv.rs      Wi-Fi setup over USB (Improv Serial)
src/discover.rs    mDNS
src/serial.rs      USB log
src/serve.rs       one-file HTTP server for local audio
src/main.rs        the havoice CLI
examples/          the examples above
```

Not affiliated with Nabu Casa, Home Assistant or ESPHome.

## License

MIT - see [LICENSE](LICENSE).
