//! Text-to-speech on this computer: macOS `say`, or `espeak-ng` / `espeak` elsewhere.
//! The result is plain PCM, which [`crate::VoicePe::say`] sends to the device as FLAC.
//!
//! `HAVOICE_TTS` picks the program, `HAVOICE_TTS_VOICE` the default voice.

use std::path::Path;

use anyhow::{Result, anyhow, bail};

use crate::audio::Pcm;

/// A voice the speech program offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Voice {
    /// What to pass as the voice: `Zuzana` for `say`, `cs` for espeak.
    pub name: String,
    /// `cs_CZ` (say) or `cs` (espeak).
    pub language: String,
    /// Sample sentence (say) or long name (espeak).
    pub about: String,
}

/// Speech program: `HAVOICE_TTS` if set (`say`, `espeak-ng`, `espeak`), else macOS `say`
/// and `espeak-ng` everywhere else.
pub fn engine() -> String {
    std::env::var("HAVOICE_TTS")
        .ok()
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| {
            if cfg!(target_os = "macos") {
                "say".into()
            } else {
                "espeak-ng".into()
            }
        })
}

fn is_say(engine: &str) -> bool {
    Path::new(engine).file_name().is_some_and(|n| n == "say")
}

/// Voice used when none is given: `HAVOICE_TTS_VOICE`, else the system default.
pub fn default_voice() -> Option<String> {
    std::env::var("HAVOICE_TTS_VOICE")
        .ok()
        .filter(|v| !v.is_empty())
}

/// Render `text` to 16-bit mono PCM in `voice` (or [`default_voice`]).
pub async fn render(text: &str, voice: Option<&str>) -> Result<Pcm> {
    let engine = engine();
    let voice = voice.map(String::from).or_else(default_voice);
    let out = std::env::temp_dir().join(format!(
        "havoice-say-{}-{}.wav",
        std::process::id(),
        rand_suffix()
    ));
    let mut cmd = tokio::process::Command::new(&engine);
    if let Some(v) = &voice {
        cmd.args(["-v", v]);
    }
    if is_say(&engine) {
        cmd.args(["--file-format=WAVE", "--data-format=LEI16@22050", "-o"]);
    } else {
        cmd.arg("-w"); // espeak: 16-bit mono WAV at 22.05 kHz
    }
    let output = cmd
        .arg(&out)
        .arg(text)
        .output()
        .await
        .map_err(|e| not_installed(&engine, e))?;
    let wav = std::fs::read(&out);
    let _ = std::fs::remove_file(&out);
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        bail!(
            "`{engine}` failed ({}){}{}",
            output.status,
            if err.trim().is_empty() { "" } else { ": " },
            err.trim()
        );
    }
    let wav = wav.map_err(|e| {
        anyhow!(
            "`{engine}` wrote no audio ({e}){}",
            voice
                .map(|v| format!(" - is {v:?} an installed voice? List them with `havoice voices`"))
                .unwrap_or_default()
        )
    })?;
    Pcm::from_wav(&wav)
}

/// Every voice the speech program offers.
pub async fn voices() -> Result<Vec<Voice>> {
    let engine = engine();
    let mut cmd = tokio::process::Command::new(&engine);
    if is_say(&engine) {
        cmd.args(["-v", "?"]);
    } else {
        cmd.arg("--voices");
    }
    let out = cmd.output().await.map_err(|e| not_installed(&engine, e))?;
    anyhow::ensure!(out.status.success(), "`{engine}` failed ({})", out.status);
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(if is_say(&engine) {
        parse_say_voices(&text)
    } else {
        parse_espeak_voices(&text)
    })
}

/// Find a voice by name (`Zuzana`, `Eddy (English (UK))`) or language (`cs`, `cs_CZ`,
/// `en-gb`), ignoring case. A language picks the first voice that speaks it.
pub async fn find_voice(query: &str) -> Result<Voice> {
    let all = voices().await?;
    pick_voice(&all, query).cloned().ok_or_else(|| {
        anyhow!(
            "no voice {query:?} in `{}` ({} voices) - list them with `havoice voices`{}",
            engine(),
            all.len(),
            if is_say(&engine()) {
                "; add more in System Settings > Accessibility > Spoken Content > System voice > Manage Voices"
            } else {
                ""
            }
        )
    })
}

fn pick_voice<'a>(all: &'a [Voice], query: &str) -> Option<&'a Voice> {
    let q = query.to_lowercase().replace('-', "_");
    let lang = |v: &Voice| v.language.to_lowercase().replace('-', "_");
    all.iter()
        .find(|v| v.name.eq_ignore_ascii_case(query))
        .or_else(|| all.iter().find(|v| lang(v) == q))
        .or_else(|| {
            all.iter()
                .find(|v| lang(v).split('_').next() == Some(q.as_str()))
        })
}

/// `say -v ?` lines: `Eddy (English (UK)) en_GB    # Hello! My name is Eddy.`
fn parse_say_voices(text: &str) -> Vec<Voice> {
    text.lines()
        .filter_map(|line| {
            let (head, about) = line.split_once('#')?;
            let mut words: Vec<&str> = head.split_whitespace().collect();
            let language = words.pop()?.to_string();
            (!words.is_empty()).then(|| Voice {
                name: words.join(" "),
                language,
                about: about.trim().to_string(),
            })
        })
        .collect()
}

/// `espeak-ng --voices` lines: ` 5  cs              --/M      Czech              zlw/cs`
fn parse_espeak_voices(text: &str) -> Vec<Voice> {
    text.lines()
        .skip(1) // header
        .filter_map(|line| {
            let cols: Vec<&str> = line.split_whitespace().collect();
            (cols.len() >= 4).then(|| Voice {
                name: cols[1].to_string(),
                language: cols[1].to_string(),
                about: cols[3].replace('_', " "),
            })
        })
        .collect()
}

fn not_installed(engine: &str, e: std::io::Error) -> anyhow::Error {
    match e.kind() {
        std::io::ErrorKind::NotFound => anyhow!(
            "speech needs `{engine}`, which isn't installed - {}",
            if cfg!(target_os = "macos") {
                "brew install espeak-ng, or unset HAVOICE_TTS to use `say`"
            } else {
                "on Ubuntu/Debian: sudo apt install espeak-ng"
            }
        ),
        _ => anyhow!(e).context(format!("running `{engine}`")),
    }
}

/// Unique-enough temp file suffix, so parallel calls don't share a file.
fn rand_suffix() -> u32 {
    let mut b = [0u8; 4];
    let _ = getrandom::fill(&mut b);
    u32::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_say_voices() {
        let v = parse_say_voices(
            "Albert              en_US    # Hello! My name is Albert.\n\
             Eddy (English (UK)) en_GB    # Hello! My name is Eddy.\n\
             Zuzana              cs_CZ    # Dobrý den, jmenuji se Zuzana.\n",
        );
        assert_eq!(v.len(), 3);
        assert_eq!(v[1].name, "Eddy (English (UK))");
        assert_eq!(v[1].language, "en_GB");
        assert_eq!(pick_voice(&v, "zuzana").unwrap().name, "Zuzana");
        assert_eq!(pick_voice(&v, "cs").unwrap().name, "Zuzana");
        assert_eq!(pick_voice(&v, "en-gb").unwrap().name, "Eddy (English (UK))");
        assert_eq!(pick_voice(&v, "en").unwrap().name, "Albert");
        assert!(pick_voice(&v, "sk").is_none());
    }

    #[test]
    fn parses_espeak_voices() {
        let v = parse_espeak_voices(
            "Pty Language       Age/Gender VoiceName          File                 Other Languages\n \
             5  cs              --/M      Czech              zlw/cs\n \
             5  en-gb           --/M      English_(Great_Britain) gmw/en\n",
        );
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].name, "cs");
        assert_eq!(v[1].about, "English (Great Britain)");
        assert_eq!(pick_voice(&v, "cs").unwrap().name, "cs");
        assert_eq!(pick_voice(&v, "en_GB").unwrap().name, "en-gb");
    }
}
