//! Generate sounds in Rust and play them on the Voice PE speaker.
//!
//!     cargo run --example synth               # all of them
//!     cargo run --example synth -- chime      # chime | arp | alarm | sweep | waves
//!
//! Each sound is plain 16-bit PCM built with `havoice::Pcm`, encoded to FLAC in-process
//! (the device firmware has no WAV decoder) and served from this Mac. The LED ring
//! takes a matching colour while it plays.

use havoice::audio::{envelope, midi, sine};
use havoice::{Pcm, Rgb, VoicePe};

const RATE: u32 = 22_050;

/// name, LED colour, generator
type Sound = (&'static str, Rgb, fn() -> Pcm);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pick = std::env::args().nth(1);
    let sounds: [Sound; 5] = [
        ("chime", Rgb(255, 200, 80), chime),
        ("arp", Rgb(120, 80, 255), arp),
        ("alarm", Rgb(255, 0, 0), alarm),
        ("sweep", Rgb(0, 200, 255), sweep),
        ("waves", Rgb(0, 80, 255), waves),
    ];
    if let Some(p) = &pick
        && !sounds.iter().any(|(n, ..)| n == p)
    {
        let names: Vec<_> = sounds.iter().map(|(n, ..)| *n).collect();
        anyhow::bail!("unknown sound {p:?}; pick one of {names:?}");
    }

    let pe = VoicePe::from_env().await?;
    for (name, colour, make) in sounds {
        if pick.as_deref().is_some_and(|p| p != name) {
            continue;
        }
        let pcm = make();
        println!("{name:6} {:.1}s", pcm.seconds());
        pe.led(colour, 0.6).await?;
        pe.play_pcm(&pcm, true).await?;
    }
    pe.led_off().await?;
    pe.close().await;
    Ok(())
}

/// Bell: a few inharmonic partials, each decaying at its own rate.
fn bell(pcm: &mut Pcm, freq: f32, len: f32) {
    pcm.push(len, |t| {
        let partials: [(f32, f32, f32); 4] = [
            (1.0, 1.0, 2.5),
            (2.0, 0.5, 4.0),
            (2.76, 0.3, 6.0),
            (5.4, 0.15, 9.0),
        ];
        let v: f32 = partials
            .iter()
            .map(|(mul, amp, decay)| amp * sine(freq * mul, t) * (-decay * t).exp())
            .sum();
        0.45 * v * envelope(t, len, 0.003, 0.05)
    });
}

/// Rising four-note bell chime.
fn chime() -> Pcm {
    let mut pcm = Pcm::new(RATE);
    for note in [72, 76, 79] {
        bell(&mut pcm, midi(note), 0.28);
    }
    bell(&mut pcm, midi(84), 1.6);
    pcm
}

/// Arpeggiated I-vi-IV-V progression with a soft square-ish voice.
fn arp() -> Pcm {
    let mut pcm = Pcm::new(RATE);
    let chords: [[u8; 4]; 4] = [
        [60, 64, 67, 72],
        [57, 60, 64, 69],
        [53, 57, 60, 65],
        [55, 59, 62, 67],
    ];
    for chord in chords {
        for &n in chord.iter().chain(chord[1..3].iter().rev()) {
            let f = midi(n + 12);
            let len = 0.14;
            pcm.push(len, |t| {
                let v = sine(f, t) + sine(3.0 * f, t) / 3.0 + sine(5.0 * f, t) / 5.0;
                0.35 * v * envelope(t, len, 0.005, 0.1)
            });
        }
    }
    bell(&mut pcm, midi(72), 1.0);
    pcm
}

/// Two-tone alarm beeps.
fn alarm() -> Pcm {
    let mut pcm = Pcm::new(RATE);
    for _ in 0..4 {
        for f in [880.0, 660.0] {
            pcm.push(0.18, |t| {
                0.4 * sine(f, t).signum() * envelope(t, 0.18, 0.005, 0.01)
            });
        }
        pcm.silence(0.08);
    }
    pcm
}

/// Exponential sweep 150 Hz -> 3 kHz and back; the phase is integrated so the pitch glides.
fn sweep() -> Pcm {
    let mut pcm = Pcm::new(RATE);
    let len = 1.5;
    for (from, to) in [(150.0f32, 3000.0f32), (3000.0, 150.0)] {
        let mut phase = 0.0f32;
        pcm.push(len, |t| {
            let f = from * (to / from).powf(t / len);
            phase += std::f32::consts::TAU * f / RATE as f32;
            0.4 * phase.sin() * envelope(t, len, 0.02, 0.02)
        });
    }
    pcm
}

/// Surf: low-passed noise whose volume swells and falls like waves.
fn waves() -> Pcm {
    let mut pcm = Pcm::new(RATE);
    let (mut seed, mut lp) = (0x2545_f491u32, 0.0f32);
    let len = 8.0;
    pcm.push(len, |t| {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let white = seed as f32 / u32::MAX as f32 * 2.0 - 1.0;
        let swell = 0.5 - 0.5 * (std::f32::consts::TAU * t / 4.0).cos(); // 4 s per wave
        lp += (0.04 + 0.25 * swell) * (white - lp); // brighter as the wave breaks
        1.6 * lp * swell * envelope(t, len, 0.5, 1.0)
    });
    pcm
}
