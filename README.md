# havoice

Rust library (+ CLI and examples) for the **Home Assistant Voice Preview Edition** (Voice PE), driven from a
Mac or a Linux machine on the same LAN. Developed and tested on macOS; see [Linux / Ubuntu](#linux--ubuntu)
for what differs there.

```rust
use havoice::{Event, Rgb, VoicePe};

let pe = VoicePe::from_env().await?;            // HAVOICE_HOST (or mDNS) + HAVOICE_KEY, reads .env
pe.led(Rgb(255, 100, 0), 0.8).await?;            // LED ring
pe.say("The laundry is done").await?;            // macOS `say` / Linux `espeak-ng` -> device speaker
pe.set_wake_word("Hey Jarvis").await?;
let mut events = pe.events().await?;
while let Some(ev) = events.next().await {
    if let Event::Button(press) = ev { println!("{press}"); }   // double_press, triple_press, long_press (single press stays with the voice assistant)
}
```

## The device (as seen from here)

| | |
|---|---|
| SoC | ESP32-S3 (16 MB flash, PSRAM) + XMOS XU316 audio DSP (fw 1.3.1, I²C `0x42`) |
| Firmware | ESPHome 2026.6.0, project *Nabu Casa.Home Assistant Voice PE* 26.9.0 |
| USB-C | ESP32-S3 native USB serial/JTAG → `/dev/cu.usbmodem3101` on macOS, `/dev/ttyACM0` on Linux (logs, Improv Wi-Fi setup, flashing) |
| Wi-Fi | `home-assistant-voice-0ae0b7.local` / `10.0.10.79` (DHCP - reserve it on the router), MAC `20:F8:3B:0A:E0:B7` |
| API | ESPHome native API on `:6053`, **Noise encrypted** (adopted by HA 2026.11.1 @ `10.0.10.112`) |
| OTA | `:3232` |
| I/O | 12× WS2812 LED ring (GRB), center button, rotary dial, mute switch, speaker + 3.5 mm out, 2 mics |
| Wake words | Okay Nabu, Hey Jarvis, Hey Mycroft (+ "Stop"), on-device microWakeWord + VAD |
| Media | speaker media player, HTTP/file sources, **Sendspin** (synced multi-room) source |
| Codecs | **FLAC and MP3 only** - the firmware has no WAV decoder, so generated audio and `say` go out as FLAC |
| Expansion | Grove port (I²C SCL/SDA; +5 V gated by GPIO46), internal headers IO1/2/38/39/40/47/48, RGBIN/RGBOUT for extra LEDs, UART0 unused |

Sources: Nabu Casa support pages (*Customizing the device*, *internal GPIO pins*, *Grove port*,
*scripts and automations*) and the device's own boot log (`logs/boot.log`).

## Build

macOS (Xcode command line tools + [rustup](https://rustup.rs)):

```bash
cargo build --release
cp .env.example .env   # then fill HAVOICE_KEY
```

Ubuntu / Debian: the `serialport` crate links `libudev`, and the HTTPS stack used by the `radio`
example compiles some C, so install these first (then the same two commands):

```bash
sudo apt install build-essential pkg-config libudev-dev cmake
sudo apt install espeak-ng      # speech (say / speak / tour / button_actions)
```

`.env` holds `HAVOICE_HOST` (IP or `.local` name; empty = first Voice PE found via mDNS) and `HAVOICE_KEY`.
Variables already set in the shell win over `.env`, so `HAVOICE_HOST=10.0.10.80 cargo run --example radio`
targets another device for one run.

## Examples

| example | needs | shows |
|---|---|---|
| `cargo run --example discover` | – | mDNS discovery + TXT records |
| `cargo run --example tour` | key | guided tour: info, entities, state snapshot, LED colours/fade/effects, speech, button/dial/mute events |
| `cargo run --example led_show [-- ff6600\|Rainbow\|off]` | key | colour wheel animation, one colour, or a firmware effect |
| `cargo run --example speak -- "text"` / `-- --file x.mp3` | key | speech (`say` / `espeak-ng`) or a local file on the speaker (as announcement) |
| `cargo run --example synth -- [chime\|arp\|alarm\|sweep\|waves]` | key | sounds generated in Rust (`havoice::Pcm`), FLAC-encoded in-process, LED colour per sound |
| `cargo run --example radio` / `-- jazz` | key | internet radio: presets, or any other argument searches radio-browser.info (MP3); now-playing titles; double press = next, triple = previous, long = stop |
| `cargo run --example button_actions` | key | desk control pad: double = says the time, triple = says the battery level, long = LED colour on/off |
| `cargo run --example pair -- --status` | USB | read-only: Wi-Fi state over USB (Improv) and whether the device has an API key |
| `cargo run --example pair [-- --wifi SSID]` | USB | no-HA setup of a factory-reset device: Wi-Fi over USB, then a fresh key into `.env` (not yet run on a real device) |

Speech (`say`, `speak`, `tour`, `button_actions`) runs a text-to-speech program on this computer and sends the
result to the device as FLAC: macOS `say`, Linux `espeak-ng`. Pick another with `HAVOICE_TTS` (`say`,
`espeak-ng`, `espeak`) and an espeak voice with `HAVOICE_TTS_VOICE` (e.g. `cs`, `en-us`).

## Library API (`havoice`)

* `VoicePe`: `connect` / `from_env`, `info`, `entities`, `led` / `led_fade` / `led_effect` / `led_off`,
  `play_url` / `play_file` / `say`, `media` / `set_volume`, `select` / `switch` / `number` / `press`,
  `wake_words` / `set_wake_word` / `set_wake_sound`, `events() -> Events` (typed `Event`s), `client()` for raw access.
* `api::Client`: lower-level client (subscribe to every raw message, send any `proto` message).
* `discover`, `serial`, `serve`: mDNS, USB log, local file server.
* `proto`: re-export of every ESPHome protobuf message (API 1.16).

## CLI

| command | needs | what |
|---|---|---|
| `havoice discover` | – | mDNS scan for ESPHome devices |
| `havoice serial [--port P] [--grep wake] [--save f]` | USB | tail the ESPHome log (opening the port does **not** reset the chip); `--port` or `HAVOICE_SERIAL`, default: the Espressif USB port (`/dev/cu.usbmodem*` / `/dev/ttyACM*`) |
| `havoice info` | key | device info |
| `havoice entities` | key | every entity, with select options / light effects / event types |
| `havoice watch` | key | live state stream: button events, dial/volume, mute, LEDs, media player |
| `havoice logs [--level debug]` | key | device log over Wi-Fi |
| `havoice led ff6600` / `led Rainbow` / `led off` | key | LED ring: colour, effect, off |
| `havoice say "hello there"` | key | speak (macOS `say`, Linux `espeak-ng`) |
| `havoice play song.mp3 [--announce] [--volume .4]` | key | streams a local file from this Mac (built-in HTTP server) or a URL |
| `havoice media pause\|stop\|volume 0.3 …` | key | media player |
| `havoice select wake_word "Hey Jarvis"` | key | any select |
| `havoice switch wake_sound off` | key | any switch |
| `havoice number <entity> <value>` / `press <entity>` | key | numbers, buttons |

Our client connects **alongside** Home Assistant (ESPHome allows several API clients), so nothing is taken away from HA.

## Linux / Ubuntu

The library, CLI and examples are plain Rust and build on Linux (packages under [Build](#build)). They have only
been run on macOS so far, so treat this section as the expected differences rather than a tested recipe.

| | macOS | Linux / Ubuntu |
|---|---|---|
| USB port | `/dev/cu.usbmodem*` | `/dev/ttyACM*` - both found automatically by Espressif's USB vendor id; override with `--port` or `HAVOICE_SERIAL` |
| USB access | works out of the box | add yourself to `dialout`: `sudo usermod -aG dialout $USER`, then log out and in |
| Speech (`say`, `speak`, `tour`, `button_actions`) | built-in `say` | `espeak-ng` (`sudo apt install espeak-ng`); a missing program is reported with that hint, and `button_actions` keeps running |
| Battery in `button_actions` | `pmset` | `/sys/class/power_supply/*/capacity` (desktops without a battery say so) |
| `.local` names | built in | Ubuntu desktop resolves them via Avahi; on a server install `apt install avahi-daemon libnss-mdns`. If the system can't resolve a `.local` name, havoice asks the device over mDNS itself |
| Firewall | System Settings → Privacy & Security → **Local Network** must allow your terminal app, or connections fail with *No route to host* | Ubuntu's `ufw` is off by default. If it's on, allow mDNS (`sudo ufw allow 5353/udp`) and let the device fetch from the built-in file server on a random port (`sudo ufw allow from 10.0.10.79`) - needed by `play <file>`, `synth`, `say` |

On Ubuntu, **ModemManager** may probe a newly plugged `/dev/ttyACM0` and send it modem commands. If the USB log
shows garbage or the port is busy right after plugging in, stop it (`sudo systemctl stop ModemManager`) or add a
udev rule setting `ENV{ID_MM_DEVICE_IGNORE}="1"` for the device.

## Getting the API key

The stock firmware requires the Noise PSK that HA generated when it adopted the device. It is **not** your HA password.

* **HA OS / Supervised**: open the *Terminal & SSH* add-on and run
  ```bash
  jq -r '.data.entries[] | select(.domain=="esphome") | "\(.title)  \(.data.host)  \(.data.noise_psk)"' /config/.storage/core.config_entries
  ```
* **HA Container**: same file in your mounted config dir.
* **ESPHome Device Builder add-on**: after *Take control*, the device YAML has `api: encryption: key:`.
* **Last resort**: flash `firmware/havoice.yaml` over USB with your own key and re-enter it in HA.

Put it in `.env` as `HAVOICE_KEY=...`. A wrong key fails fast with `Handshake MAC failure`.

## Layout

```
src/lib.rs              crate root, re-exports
src/device.rs           VoicePe high-level API + typed Events
src/api.rs              connection over esphome-client: reader task, broadcast, entities, error hints
src/discover.rs         mDNS (esphome_client::discovery) + IPv4 lookup of .local names
src/serial.rs           USB log tail
src/improv.rs           Improv Wi-Fi setup over the USB serial port
src/audio.rs            Pcm samples, FLAC encoding, WAV reading
src/serve.rs            one-file HTTP server for play_file / say
src/main.rs             `havoice` CLI on top of the library
examples/               runnable demos (see above)
tests/fake_device.rs    end-to-end tests against a fake encrypted device (esphome-native-api device side)
tests/pairing.rs        pairing a key-less fake device
firmware/havoice.yaml   custom firmware stub (not flashed)
```

Transport, Noise NNpsk0 handshake and protobuf messages come from the
[`esphome-client`](https://crates.io/crates/esphome-client) crate (pinned to API 1.16). We skip its built-in
connection setup and send Hello ourselves to keep the `HelloResponse`, and skip messages newer than its proto set.
The tests run our client against the *device* side of `esphome-native-api`, so two independent implementations
have to agree.

## Ideas to build next

1. **Status light for the Mac**: drive the ring from scripts, e.g. a Claude Code hook that turns it blue while
   waiting for input, red on failed builds, green when CI passes.
2. **Desk macro pad**: map `watch` events (double/triple/long press, dial) to Mac actions such as
   play/pause, Zoom mute, or volume.
3. **Mac → device speech**: `say -o x.aiff` → `afconvert` → `havoice play --announce` for spoken notifications.
4. **Own voice pipeline in Rust**: the API's voice-assistant messages stream mic audio after the wake word; plug in
   whisper → Claude → TTS and send audio back. Catch: only one client may own the voice assistant (currently HA),
   so either take it over temporarily or do it inside HA as a custom conversation agent.
5. **Custom firmware**: Grove sensors (temp/CO₂/PIR/light), extra WS2812 strip on RGBOUT, custom LED animations,
   raw button/dial events. Built and flashed over USB from `firmware/`.
6. **Sendspin**: the firmware already has a Sendspin source, so a Rust Sendspin server could stream synced audio from the Mac.
