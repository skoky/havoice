//! mDNS discovery of ESPHome devices, via `esphome_client::discovery`.

use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, Ipv4Addr};
use std::time::{Duration, Instant};

use anyhow::Result;
use esphome_client::discovery;
use mdns_sd::{HostnameResolutionEvent, ServiceDaemon};

/// An ESPHome device announced on the LAN.
#[derive(Debug, Clone)]
pub struct Found {
    /// Device name, e.g. `home-assistant-voice-0a1b2c`.
    pub name: String,
    /// mDNS hostname, e.g. `home-assistant-voice-0a1b2c.local`.
    pub host: String,
    /// API port (6053).
    pub port: u16,
    /// Whether the API needs a key (Noise encryption).
    pub encrypted: bool,
    /// TXT record: `version`, `mac`, `friendly_name`, `project_version`,
    /// `package_import_url`, ...
    pub txt: HashMap<String, String>,
}

/// Every ESPHome device that answers within `wait` (3 s is plenty on a LAN).
pub async fn discover(wait: Duration) -> Result<Vec<Found>> {
    let mut stream = discovery::Client::default().discover()?;
    let mut found: BTreeMap<String, Found> = BTreeMap::new();
    let end = Instant::now() + wait;
    while let Some(left) = end.checked_duration_since(Instant::now()) {
        let Ok(Ok(d)) = tokio::time::timeout(left, stream.next()).await else {
            break;
        };
        let host = d.hostname().trim_end_matches('.').to_string();
        let name = host.trim_end_matches(".local").to_string();
        found.insert(
            name.clone(),
            Found {
                name,
                host,
                port: d.socket_address().map_or(6053, |a| a.port()),
                encrypted: d.has_encryption(),
                txt: d.attributes(),
            },
        );
    }
    Ok(found.into_values().collect())
}

/// IPv4 address of a `.local` name (e.g. `home-assistant-voice-0a1b2c.local`), asked over
/// mDNS directly. macOS's resolver sometimes has only the device's IPv6 link-local address
/// cached and answers with that alone; this asks the device itself.
pub async fn resolve_v4(host: &str, wait: Duration) -> Result<Option<Ipv4Addr>> {
    let mdns = ServiceDaemon::new()?;
    let name = format!("{}.", host.trim_end_matches('.'));
    let rx = mdns.resolve_hostname(&name, Some(wait.as_millis() as u64))?;
    let search = async {
        while let Ok(ev) = rx.recv_async().await {
            match ev {
                HostnameResolutionEvent::AddressesFound(_, addrs) => {
                    let v4 = addrs.iter().find_map(|a| match a.to_ip_addr() {
                        IpAddr::V4(ip) => Some(ip),
                        IpAddr::V6(_) => None,
                    });
                    if v4.is_some() {
                        return v4;
                    }
                }
                HostnameResolutionEvent::SearchTimeout(_)
                | HostnameResolutionEvent::SearchStopped(_) => return None,
                _ => {}
            }
        }
        None
    };
    // the daemon times out by itself; this only guards against it never answering
    let found = tokio::time::timeout(wait + Duration::from_secs(1), search)
        .await
        .ok()
        .flatten();
    let _ = mdns.shutdown();
    Ok(found)
}
