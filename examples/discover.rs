//! Find ESPHome devices on the LAN. No key needed.
//!
//!     cargo run --example discover

use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let found = havoice::discover::discover(Duration::from_secs(3)).await?;
    if found.is_empty() {
        eprintln!(
            "no ESPHome devices answered within 3 s. Is the device on Wi-Fi \
             (`ping` it, or `cargo run --example pair -- --status` over USB)? \
             macOS can also block this app: System Settings > Privacy & Security > Local Network."
        );
    }
    for d in found {
        println!(
            "{} ({}:{}) encrypted={}",
            d.name, d.host, d.port, d.encrypted
        );
        let mut txt: Vec<_> = d.txt.iter().collect();
        txt.sort();
        for (k, v) in txt {
            println!("    {k} = {v}");
        }
    }
    Ok(())
}
