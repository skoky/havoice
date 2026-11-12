//! Improv Wi-Fi over the USB serial port (<https://www.improv-wifi.com/serial/>).
//!
//! Gives a freshly reset device its Wi-Fi credentials without Home Assistant, a phone or
//! Bluetooth - the same thing ESP Web Tools does in the browser. The port also carries
//! the device log, so packets are fished out of the byte stream by their `IMPROV` header.

use std::io::{Read, Write};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use serialport::SerialPort;

const HEADER: &[u8] = b"IMPROV";
const VERSION: u8 = 1;

/// Provisioning state reported by the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Stopped,
    AwaitingAuthorization,
    /// ready to accept credentials
    Ready,
    Provisioning,
    /// connected to Wi-Fi
    Provisioned,
    Unknown(u8),
}

impl From<u8> for State {
    fn from(b: u8) -> Self {
        match b {
            0x00 => Self::Stopped,
            0x01 => Self::AwaitingAuthorization,
            0x02 => Self::Ready,
            0x03 => Self::Provisioning,
            0x04 => Self::Provisioned,
            other => Self::Unknown(other),
        }
    }
}

/// A Wi-Fi network the device can see.
#[derive(Debug, Clone)]
pub struct Network {
    pub ssid: String,
    pub rssi: i32,
    pub secured: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Packet {
    State(u8),
    Error(u8),
    Result { command: u8, strings: Vec<String> },
}

const CMD_WIFI: u8 = 0x01;
const CMD_STATE: u8 = 0x02;
const CMD_INFO: u8 = 0x03;
const CMD_SCAN: u8 = 0x04;

pub struct Improv {
    port: Box<dyn SerialPort>,
    buf: Vec<u8>,
}

impl Improv {
    /// Open the device's USB serial port (see [`crate::serial::default_port`]).
    pub fn open(path: &str) -> Result<Self> {
        // default open asserts DTR+RTS, which does *not* reset the ESP32-S3 (see serial.rs)
        let port = serialport::new(path, 115_200)
            .timeout(Duration::from_millis(100))
            .open()
            .with_context(|| format!("opening {path}"))?;
        Ok(Self {
            port,
            buf: Vec::new(),
        })
    }

    /// Current state, plus the device URL once it's on Wi-Fi.
    pub fn state(&mut self) -> Result<(State, Option<String>)> {
        self.rpc(CMD_STATE, &[])?;
        let mut state = None;
        let until = Instant::now() + Duration::from_secs(3);
        while let Some(p) = self.recv(until)? {
            match p {
                Packet::State(s) => {
                    state = Some(State::from(s));
                    if state != Some(State::Provisioned) {
                        break; // a URL only follows when provisioned
                    }
                }
                Packet::Result { command, strings } if command == CMD_STATE => {
                    return Ok((State::Provisioned, strings.into_iter().next()));
                }
                _ => {}
            }
        }
        state
            .map(|s| (s, None))
            .ok_or_else(|| anyhow!("no Improv reply - is improv_serial in the firmware?"))
    }

    /// Firmware name, version, chip, device name (and OS name/version if any).
    pub fn info(&mut self) -> Result<Vec<String>> {
        self.rpc(CMD_INFO, &[])?;
        self.result(CMD_INFO, Duration::from_secs(3))
    }

    /// Networks the device sees, strongest first.
    pub fn scan(&mut self) -> Result<Vec<Network>> {
        self.rpc(CMD_SCAN, &[])?;
        let mut out = Vec::new();
        loop {
            let s = self.result(CMD_SCAN, Duration::from_secs(15))?;
            if s.len() < 3 {
                break; // empty result terminates the list
            }
            out.push(Network {
                ssid: s[0].clone(),
                rssi: s[1].parse().unwrap_or(-100),
                secured: s[2] == "YES",
            });
        }
        out.sort_by_key(|n| -n.rssi);
        out.dedup_by(|a, b| a.ssid == b.ssid);
        Ok(out)
    }

    /// Send Wi-Fi credentials and wait until the device has joined.
    /// Returns the device URL (e.g. `http://10.0.10.76`), if it reports one.
    pub fn provision(
        &mut self,
        ssid: &str,
        password: &str,
        timeout: Duration,
    ) -> Result<Option<String>> {
        ensure!(
            ssid.len() < 128 && password.len() < 128,
            "SSID/password too long"
        );
        let mut data = vec![ssid.len() as u8];
        data.extend(ssid.as_bytes());
        data.push(password.len() as u8);
        data.extend(password.as_bytes());
        self.rpc(CMD_WIFI, &data)?;

        let until = Instant::now() + timeout;
        while let Some(p) = self.recv(until)? {
            match p {
                Packet::Error(0) => {}
                Packet::Error(3) => bail!("device could not join {ssid:?} - wrong password?"),
                Packet::Error(e) => bail!("Improv error {e:#04x}"),
                Packet::Result { command, strings } if command == CMD_WIFI => {
                    return Ok(strings.into_iter().find(|s| !s.is_empty()));
                }
                Packet::State(s) if State::from(s) == State::Provisioned => {
                    // the URL result usually follows right away
                    let soon = Instant::now() + Duration::from_secs(2);
                    while let Some(p) = self.recv(soon)? {
                        if let Packet::Result { strings, .. } = p {
                            return Ok(strings.into_iter().find(|s| !s.is_empty()));
                        }
                    }
                    return Ok(None);
                }
                _ => {}
            }
        }
        bail!("device did not join Wi-Fi within {timeout:?}")
    }

    fn rpc(&mut self, command: u8, data: &[u8]) -> Result<()> {
        let mut body = vec![command, data.len() as u8];
        body.extend(data);
        self.port.write_all(&frame(0x03, &body))?;
        self.port.flush()?;
        Ok(())
    }

    fn result(&mut self, command: u8, wait: Duration) -> Result<Vec<String>> {
        let until = Instant::now() + wait;
        while let Some(p) = self.recv(until)? {
            match p {
                Packet::Result {
                    command: c,
                    strings,
                } if c == command => return Ok(strings),
                Packet::Error(0) | Packet::State(_) => {}
                Packet::Error(e) => bail!("Improv error {e:#04x}"),
                Packet::Result { .. } => {}
            }
        }
        bail!("no Improv reply to command {command:#04x}")
    }

    /// Next packet before `until`, skipping log text in between.
    fn recv(&mut self, until: Instant) -> Result<Option<Packet>> {
        loop {
            if let Some(p) = take_packet(&mut self.buf) {
                return Ok(Some(p));
            }
            if Instant::now() >= until {
                return Ok(None);
            }
            let mut chunk = [0u8; 1024];
            match self.port.read(&mut chunk) {
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
}

/// `IMPROV | version | type | len | data | checksum` + newline.
fn frame(kind: u8, data: &[u8]) -> Vec<u8> {
    let mut f = HEADER.to_vec();
    f.extend([VERSION, kind, data.len() as u8]);
    f.extend(data);
    let sum = f.iter().fold(0u8, |a, &b| a.wrapping_add(b));
    f.push(sum);
    f.push(b'\n');
    f
}

/// Pull the first complete, valid packet out of `buf`, dropping anything before it.
fn take_packet(buf: &mut Vec<u8>) -> Option<Packet> {
    loop {
        let start = buf.windows(HEADER.len()).position(|w| w == HEADER)?;
        buf.drain(..start);
        if buf.len() < 9 {
            return None;
        }
        let len = buf[8] as usize;
        if buf.len() < 9 + len + 1 {
            return None;
        }
        let sum = buf[..9 + len].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        if buf[6] != VERSION || buf[9 + len] != sum {
            buf.drain(..1); // false match inside log text - keep looking
            continue;
        }
        let kind = buf[7];
        let data: Vec<u8> = buf[9..9 + len].to_vec();
        buf.drain(..9 + len + 1);
        return Some(match kind {
            0x01 => Packet::State(*data.first()?),
            0x02 => Packet::Error(*data.first()?),
            0x04 => Packet::Result {
                command: *data.first()?,
                strings: strings(data.get(2..).unwrap_or_default()),
            },
            _ => continue,
        });
    }
}

fn strings(mut d: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    while let Some((&n, rest)) = d.split_first() {
        let n = (n as usize).min(rest.len());
        out.push(String::from_utf8_lossy(&rest[..n]).into_owned());
        d = &rest[n..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wifi_frame_matches_spec_example() {
        // spec: SSID MyWirelessAP, password mysecurepassword -> 01 1E 0C ... 10 ...
        let mut body = vec![0x01, 0x1e, 0x0c];
        body.extend(b"MyWirelessAP");
        body.push(0x10);
        body.extend(b"mysecurepassword");
        let f = frame(0x03, &body);
        assert_eq!(&f[..9], b"IMPROV\x01\x03\x20");
        let sum = f[..f.len() - 2].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        assert_eq!(f[f.len() - 2], sum);
    }

    #[test]
    fn packets_are_found_between_log_lines() {
        let mut result = vec![CMD_STATE, 0];
        result.push(17);
        result.extend(b"http://10.0.10.76");
        result[1] = (result.len() - 2) as u8;
        let mut buf = b"[D][wifi]: noise IMPRO\n".to_vec();
        buf.extend(&frame(0x01, &[0x04])[..11]); // drop the newline
        buf.extend(b"[I][app]: more log\n");
        buf.extend(frame(0x04, &result));
        assert_eq!(take_packet(&mut buf), Some(Packet::State(4)));
        assert_eq!(
            take_packet(&mut buf),
            Some(Packet::Result {
                command: CMD_STATE,
                strings: vec!["http://10.0.10.76".into()]
            })
        );
        assert_eq!(take_packet(&mut buf), None);
    }
}
