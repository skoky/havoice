//! Tiny one-file HTTP server so the device can stream a local audio file from this Mac.

use std::net::{IpAddr, UdpSocket};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// The local IP this Mac uses to reach `device_ip`.
pub fn lan_ip_towards(device_ip: IpAddr) -> Result<IpAddr> {
    let s = UdpSocket::bind("0.0.0.0:0")?;
    s.connect((device_ip, 6053))?;
    Ok(s.local_addr()?.ip())
}

fn mime(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("mp3") => "audio/mpeg",
        Some("flac") => "audio/flac",
        Some("wav") => "audio/wav",
        Some("ogg") | Some("opus") => "audio/ogg",
        _ => "application/octet-stream",
    }
}

/// Serve `file` on an ephemeral port; returns the URL the device should fetch.
pub async fn serve_file(file: PathBuf, device_ip: IpAddr) -> Result<String> {
    let body = tokio::fs::read(&file)
        .await
        .with_context(|| format!("reading {}", file.display()))?;
    let name = file.file_name().unwrap_or_default().to_string_lossy();
    serve_bytes(body, &name, device_ip).await
}

/// Serve in-memory `body` as `name` (the extension picks the Content-Type, which the
/// device uses to choose a decoder) on an ephemeral port; returns the URL.
pub async fn serve_bytes(body: Vec<u8>, name: &str, device_ip: IpAddr) -> Result<String> {
    let listener = TcpListener::bind("0.0.0.0:0").await?;
    let port = listener.local_addr()?.port();
    let ip = lan_ip_towards(device_ip)?;
    let ctype = mime(Path::new(name));
    let name = name.replace(' ', "%20");
    let body = std::sync::Arc::new(body);
    tokio::spawn(async move {
        while let Ok((mut sock, peer)) = listener.accept().await {
            let body = body.clone();
            tokio::spawn(async move {
                let mut req = [0u8; 2048];
                let _ = sock.read(&mut req).await;
                eprintln!("# {peer} fetched the file");
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(hdr.as_bytes()).await;
                let _ = sock.write_all(&body).await;
            });
        }
    });
    Ok(format!("http://{ip}:{port}/{name}"))
}
