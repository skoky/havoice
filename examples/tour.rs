//! A guided tour of what the Voice PE can do from Rust.
//!
//!     cargo run --example tour
//!
//! Needs HAVOICE_KEY (and optionally HAVOICE_HOST) in the env or `.env`.
//! Touches: LED ring, speaker volume, speech. Restores volume and turns the ring off at the end.

use std::time::Duration;

use havoice::{Event, Rgb, VoicePe};
use tokio::time::sleep;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Connect: mDNS finds the device, the Noise key authenticates us next to Home Assistant.
    let pe = VoicePe::from_env().await?;
    let info = pe.info().await?;
    println!(
        "connected to {} - {} {}",
        info.friendly_name, info.project_name, info.project_version
    );
    println!(
        "  ESPHome {} on {}, MAC {}",
        info.esphome_version, info.model, info.mac_address
    );

    // 2. What's on it.
    println!("\nentities:");
    for e in pe.entities() {
        let extra = if e.choices.is_empty() {
            String::new()
        } else {
            format!("  {:?}", e.choices)
        };
        println!("  {:14} {:36}{extra}", e.kind, e.object_id);
    }

    // 3. Current state (first batch of events is a snapshot of everything).
    let mut events = pe.events().await?;
    let mut volume = None;
    while let Ok(Some(ev)) = tokio::time::timeout(Duration::from_millis(800), events.next()).await {
        if let Event::Media { volume: v, .. } = ev {
            volume = Some(v);
        }
        println!("  state: {ev:?}");
    }

    // 4. LED ring: a few colours, a fade, then the built-in effects.
    println!("\nLED ring ...");
    for (name, c) in [
        ("red", Rgb(255, 0, 0)),
        ("green", Rgb(0, 255, 0)),
        ("blue", Rgb(0, 0, 255)),
    ] {
        println!("  {name}");
        pe.led(c, 0.6).await?;
        sleep(Duration::from_millis(700)).await;
    }
    println!("  fade to warm white");
    pe.led_fade(Rgb(255, 160, 60), 0.4, 2.0).await?;
    sleep(Duration::from_secs(2)).await;
    for effect in pe.led_effects().into_iter().filter(|e| e != "None").take(3) {
        println!("  effect {effect}");
        pe.led_effect(&effect).await?;
        sleep(Duration::from_secs(3)).await;
    }

    // 5. Speaker: talk through it with the Mac's voice.
    println!("\nspeaking ...");
    pe.set_volume(0.4).await?;
    pe.say("Hello! I am the Home Assistant Voice, now remote controlled from Rust.")
        .await?;

    // 6. Settings you can flip: wake words, wake sound.
    println!("\nwake words available: {:?}", pe.wake_words());

    // 7. Input: listen to the button / dial / mute switch for 15 s.
    println!("\nnow press the button, turn the dial or flip mute (15 s) ...");
    let until = tokio::time::Instant::now() + Duration::from_secs(15);
    while let Ok(Some(ev)) = tokio::time::timeout_at(until, events.next()).await {
        match ev {
            Event::Button(press) => {
                println!("  button: {press}");
                pe.led(Rgb::wheel(rand_byte()), 0.8).await?;
            }
            Event::Media { volume, muted, .. } => {
                println!("  volume {:.0}% muted={muted}", volume * 100.0)
            }
            Event::Switch { name, on } => println!("  switch {name} = {on}"),
            Event::Light { .. } => {} // our own LED changes echo back
            other => println!("  {other:?}"),
        }
    }

    // Clean up.
    pe.led_off().await?;
    if let Some(v) = volume {
        pe.set_volume(v).await?;
    }
    pe.close().await;
    println!("\ndone.");
    Ok(())
}

fn rand_byte() -> u8 {
    (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos()
        >> 8) as u8
}
