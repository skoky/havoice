//! Use the Voice PE as a desk control: map button presses to actions on the Mac.
//!
//!     cargo run --example button_actions
//!
//! double press -> speak the time
//! triple press -> speak the Mac's battery level
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
                        pe.say(&format!("It is {now}")).await?;
                    }
                    "triple_press" => pe.say(&battery()).await?,
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

fn battery() -> String {
    let out = std::process::Command::new("pmset")
        .args(["-g", "batt"])
        .output();
    let text = out
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    match regex::Regex::new(r"(\d+)%").unwrap().captures(&text) {
        Some(c) => format!("The Mac battery is at {} percent", &c[1]),
        None => "This Mac has no battery".into(),
    }
}
