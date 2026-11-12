//! Speak text on the Voice PE using the Mac's `say` voice (played as an announcement,
//! so music ducks underneath).
//!
//!     cargo run --example speak -- "The laundry is done"
//!     cargo run --example speak -- --file ~/Music/jingle.mp3

use havoice::VoicePe;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pe = VoicePe::from_env().await?;
    match args.as_slice() {
        [flag, path] if flag == "--file" => {
            let url = pe.play_file(path, true).await?;
            println!("playing {url} (Ctrl-C when it's done)");
            tokio::signal::ctrl_c().await?;
        }
        [] => pe.say("Hello from the Mac").await?,
        words => pe.say(&words.join(" ")).await?,
    }
    pe.close().await;
    Ok(())
}
