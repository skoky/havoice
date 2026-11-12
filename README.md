# havoice

Rust library (+ CLI and examples) for the **Home Assistant Voice Preview Edition** (Voice PE) plugged into this Mac.

```rust
use havoice::{Event, Rgb, VoicePe};

let pe = VoicePe::from_env().await?;            // HAVOICE_HOST (or mDNS) + HAVOICE_KEY, reads .env
pe.led(Rgb(255, 100, 0), 0.8).await?;            // LED ring
pe.say("The laundry is done").await?;            // macOS `say` -> device speaker
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
| USB-C | ESP32-S3 native USB serial/JTAG → `/dev/cu.usbmodem3101` (logs + flashing) |
| Wi-Fi | `home-assistant-voice-0ae0b7.local` / `10.0.10.76`, MAC `20:F8:3B:0A:E0:B7` |
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

```bash
cargo build --release
cp .env.example .env   # then fill HAVOICE_KEY
```

## Examples

| example | needs | shows |
|---|---|---|
| `cargo run --example discover` | – | mDNS discovery + TXT records |
| `cargo run --example tour` | key | guided tour: info, entities, state snapshot, LED colours/fade/effects, speech, button/dial/mute events |
| `cargo run --example led_show [-- ff6600\|Rainbow\|off]` | key | colour wheel animation, one colour, or a firmware effect |
| `cargo run --example speak -- "text"` / `-- --file x.mp3` | key | Mac speech or a local file on the speaker (as announcement) |
| `cargo run --example synth -- [chime\|arp\|alarm\|sweep\|waves]` | key | sounds generated in Rust (`havoice::Pcm`), FLAC-encoded in-process, LED colour per sound |
| `cargo run --example radio` / `-- jazz` | key | internet radio: presets or a radio-browser.info search (MP3), now-playing titles; double press = next, triple = previous, long = stop |
| `cargo run --example button_actions` | key | desk control pad: double = says the time, triple = Mac battery, long = LED colour on/off |

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
| `havoice serial [--grep wake] [--save f]` | USB | tail the ESPHome log (opening the port does **not** reset the chip) |
| `havoice info` | key | device info |
| `havoice entities` | key | every entity, with select options / light effects / event types |
| `havoice watch` | key | live state stream: button events, dial/volume, mute, LEDs, media player |
| `havoice logs [--level debug]` | key | device log over Wi-Fi |
| `havoice led ff6600` / `led Rainbow` / `led off` | key | LED ring: colour, effect, off |
| `havoice say "hello there"` | key | speak with the Mac's voice |
| `havoice play song.mp3 [--announce] [--volume .4]` | key | streams a local file from this Mac (built-in HTTP server) or a URL |
| `havoice media pause\|stop\|volume 0.3 …` | key | media player |
| `havoice select wake_word "Hey Jarvis"` | key | any select |
| `havoice switch wake_sound off` | key | any switch |
| `havoice number <entity> <value>` / `press <entity>` | key | numbers, buttons |

Our client connects **alongside** Home Assistant (ESPHome allows several API clients), so nothing is taken away from HA.

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
src/discover.rs         mDNS (esphome_client::discovery)
src/serial.rs           USB log tail
src/serve.rs            one-file HTTP server for play_file / say
src/main.rs             `havoice` CLI on top of the library
examples/               runnable demos (see above)
tests/fake_device.rs    end-to-end tests against a fake encrypted device (esphome-native-api device side)
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
