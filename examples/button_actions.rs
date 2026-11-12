//! Use the Voice PE as a desk control: map button presses to actions on this computer.
//!
//!     cargo run --example button_actions
//!
//! double press -> speak the time
//! triple press -> speak this computer's battery level (macOS `pmset`, Linux /sys/class/power_supply)
//! long press   -> LED on (new colour) / off
//! dial / mute  -> printed
//!
//! (A single press isn't reported over the API: the firmware keeps it for the voice assistant.)
//!
//! Home Assistant still gets every press too; this only listens alongside it.

use havoice::{Event, Rgb, VoicePe};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pe = VoicePe::from_env().await?;
    let mut events = pe.events().await?;
    let mut hue: u8 = 0;
    let mut lit = false;
    let mut snapshot_done = false;
    eprintln!("listening - press the centre button (Ctrl-C to stop)");

    while let Some(ev) = events.next().await {
        match ev {
            Event::Button(press) => {
                snapshot_done = true;
                println!("button: {press}");
                match press.as_str() {
                    "double_press" => {
                        let now = chrono::Local::now().format("%H:%M").to_string();
                        speak(&pe, &format!("It is {now}")).await;
                    }
                    "triple_press" => speak(&pe, &battery()).await,
                    "long_press" => {
                        lit = !lit;
                        if lit {
                            hue = hue.wrapping_add(53);
                            pe.led(Rgb::wheel(hue), 0.8).await?;
                        } else {
                            pe.led_off().await?;
                        }
                    }
                    _ => {}
                }
            }
            Event::Media {
                volume,
                muted,
                state,
            } if snapshot_done => {
                println!("media: {state} volume {:.0}% muted={muted}", volume * 100.0)
            }
            Event::Switch { name, on } if snapshot_done => println!("switch {name}: {on}"),
            _ => {}
        }
    }
    eprintln!("connection closed: {}", pe.client().close_reason());
    Ok(())
}

/// Say `text`, but keep listening if speech fails (e.g. no espeak-ng on Linux).
async fn speak(pe: &VoicePe, text: &str) {
    println!("  says: {text}");
    if let Err(e) = pe.say(text).await {
        eprintln!("  speech failed: {e:#}");
    }
}

fn battery() -> String {
    match battery_percent() {
        Some(p) => format!("The battery is at {p} percent"),
        None => "This computer has no battery".into(),
    }
}

#[cfg(target_os = "macos")]
fn battery_percent() -> Option<u32> {
    let out = std::process::Command::new("pmset")
        .args(["-g", "batt"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    regex::Regex::new(r"(\d+)%").unwrap().captures(&text)?[1]
        .parse()
        .ok()
}

#[cfg(not(target_os = "macos"))]
fn battery_percent() -> Option<u32> {
    // Linux: the first power supply of type Battery
    std::fs::read_dir("/sys/class/power_supply")
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| std::fs::read_to_string(p.join("type")).is_ok_and(|t| t.trim() == "Battery"))
        .and_then(|p| std::fs::read_to_string(p.join("capacity")).ok())
        .and_then(|c| c.trim().parse().ok())
}
