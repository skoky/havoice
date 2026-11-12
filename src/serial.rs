//! Tail the ESPHome logger on the ESP32-S3 native USB serial/JTAG port.

use std::io::{Read, Write};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use regex::Regex;

/// Espressif's USB vendor id - the ESP32-S3's built-in USB serial/JTAG.
const ESPRESSIF_VID: u16 = 0x303a;

/// The device's USB serial port: `/dev/cu.usbmodem*` on macOS, `/dev/ttyACM*` on Linux.
/// An Espressif chip wins over other USB modems.
pub fn default_port() -> Result<String> {
    let ports: Vec<_> = serialport::available_ports()?
        .into_iter()
        .map(|p| {
            let espressif = matches!(
                &p.port_type,
                serialport::SerialPortType::UsbPort(u) if u.vid == ESPRESSIF_VID
            );
            (p.port_name, espressif)
        })
        .collect();
    pick_port(ports).ok_or_else(|| {
        anyhow!(
            "no USB serial port found (/dev/cu.usbmodem* on macOS, /dev/ttyACM* on Linux) - \
             is the device plugged in via USB-C? Or pass the port explicitly"
        )
    })
}

/// Best candidate from (name, is-Espressif) pairs. macOS lists every port twice, as
/// `/dev/tty.*` (blocks on open) and `/dev/cu.*`; only the `cu` one is usable here.
fn pick_port(ports: Vec<(String, bool)>) -> Option<String> {
    let mut candidates: Vec<_> = ports
        .into_iter()
        .filter(|(n, _)| n.starts_with("/dev/cu.usbmodem") || n.starts_with("/dev/ttyACM"))
        .collect();
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    candidates.into_iter().next().map(|(n, _)| n)
}

/// Print the device log from `port` to stdout until the port closes (blocking).
/// `grep` keeps only matching lines, `raw` keeps ANSI colours, `save` also appends the
/// plain lines to a file. Opening the port does not reset the chip.
pub fn tail(
    port: &str,
    grep: Option<&Regex>,
    raw: bool,
    mut save: Option<std::fs::File>,
) -> Result<()> {
    let ansi = Regex::new(r"\x1b\[[0-9;]*m")?;
    // ESP32-S3 USB-JTAG resets on RTS=1 && DTR=0. macOS asserts both on open, which is safe,
    // so keep DTR on (dtr_on_open(false) would cause exactly the reset state).
    let mut sp = serialport::new(port, 115_200)
        .timeout(Duration::from_millis(200))
        .open()
        .with_context(|| format!("opening {port}"))?;
    eprintln!("# {port} - Ctrl-C to stop");
    let mut buf = Vec::<u8>::new();
    let mut chunk = [0u8; 4096];
    let stdout = std::io::stdout();
    loop {
        match sp.read(&mut chunk) {
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => continue,
            Err(e) => return Err(e).context("serial read failed (device unplugged/rebooted?)"),
        }
        while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\r', '\n']);
            let plain = ansi.replace_all(line, "");
            if grep.is_some_and(|g| !g.is_match(&plain)) {
                continue;
            }
            let mut out = stdout.lock();
            writeln!(out, "{}", if raw { line } else { &plain })?;
            if let Some(f) = save.as_mut() {
                writeln!(f, "{plain}")?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::pick_port;

    fn ports(list: &[(&str, bool)]) -> Vec<(String, bool)> {
        list.iter().map(|(n, e)| (n.to_string(), *e)).collect()
    }

    #[test]
    fn macos_takes_the_cu_device() {
        let p = ports(&[
            ("/dev/tty.usbmodem3101", true),
            ("/dev/cu.Bluetooth-Incoming-Port", false),
            ("/dev/cu.usbmodem3101", true),
        ]);
        assert_eq!(pick_port(p).as_deref(), Some("/dev/cu.usbmodem3101"));
    }

    #[test]
    fn linux_prefers_the_espressif_chip() {
        let p = ports(&[
            ("/dev/ttyS0", false),
            ("/dev/ttyACM0", false), // some other USB modem
            ("/dev/ttyACM1", true),
        ]);
        assert_eq!(pick_port(p).as_deref(), Some("/dev/ttyACM1"));
        assert_eq!(pick_port(ports(&[("/dev/ttyS0", false)])), None);
    }
}
