//! Colour cycle on the LED ring, or a single colour/effect.
//!
//!     cargo run --example led_show               # rainbow cycle for 10 s
//!     cargo run --example led_show -- ff6600     # one colour
//!     cargo run --example led_show -- Rainbow    # a firmware effect (see `tour` for the list)
//!     cargo run --example led_show -- off

use std::time::Duration;

use havoice::{Rgb, VoicePe};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pe = VoicePe::from_env().await?;
    match std::env::args().nth(1).as_deref() {
        Some("off") => pe.led_off().await?,
        Some(arg) if Rgb::hex(arg).is_ok() && arg.trim_start_matches('#').len() == 6 => {
            pe.led(Rgb::hex(arg)?, 0.8).await?
        }
        Some(effect) => {
            let effects = pe.led_effects();
            anyhow::ensure!(
                effects.iter().any(|e| e == effect),
                "unknown effect {effect:?}; have {effects:?}"
            );
            pe.led_effect(effect).await?;
        }
        None => {
            for step in 0..100u32 {
                pe.led(Rgb::wheel((step * 8 % 256) as u8), 0.7).await?;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            pe.led_off().await?;
        }
    }
    pe.close().await;
    Ok(())
}
