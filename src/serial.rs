//! Tail the ESPHome logger on the ESP32-S3 native USB serial/JTAG port.

use std::io::{Read, Write};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use regex::Regex;

pub fn default_port() -> Result<String> {
    let mut ports: Vec<String> = serialport::available_ports()?
        .into_iter()
        .map(|p| p.port_name)
        .filter(|n| n.starts_with("/dev/cu.usbmodem"))
        .collect();
    ports.sort();
    match ports.into_iter().next() {
        Some(p) => Ok(p),
        None => bail!("no /dev/cu.usbmodem* found - is the device plugged in via USB-C?"),
    }
}

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
