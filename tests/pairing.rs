//! Pairing without Home Assistant: a fake device first speaks plaintext (no key yet), takes
//! the key we send, then comes back encrypted with exactly that key.
//!
//! `esphome-native-api` can't parse NoiseEncryptionSetKeyRequest, so the plaintext half
//! is a tiny hand-written device using esphome-client's message codec.

use esphome_client::types::{self as pb, EspHomeMessage as Msg};
use esphome_native_api::esphomeapi::EspHomeApi;
use esphome_native_api::parser::ProtoMessage;
use esphome_native_api::proto as npb;
use havoice::VoicePe;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

async fn read_varint(s: &mut TcpStream) -> std::io::Result<usize> {
    let (mut v, mut shift) = (0usize, 0);
    loop {
        let b = s.read_u8().await?;
        v |= ((b & 0x7f) as usize) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
    }
}

fn varint(mut v: usize, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Plaintext frame: 0x00 | varint len | varint type | protobuf.
async fn read_msg(s: &mut TcpStream) -> std::io::Result<Msg> {
    assert_eq!(
        s.read_u8().await?,
        0,
        "client must speak plaintext to a key-less device"
    );
    let len = read_varint(s).await?;
    let kind = read_varint(s).await? as u16;
    let mut payload = vec![0; len];
    s.read_exact(&mut payload).await?;
    let raw = [
        &kind.to_be_bytes()[..],
        &(len as u16).to_be_bytes(),
        &payload,
    ]
    .concat();
    Ok(Msg::try_from(raw).expect("decodable message"))
}

async fn write_msg(s: &mut TcpStream, m: impl Into<Msg>) {
    let raw: Vec<u8> = m.into().into(); // u16 type | u16 len | payload
    let kind = u16::from_be_bytes([raw[0], raw[1]]) as usize;
    let mut out = vec![0];
    varint(raw.len() - 4, &mut out);
    varint(kind, &mut out);
    out.extend(&raw[4..]);
    s.write_all(&out).await.unwrap();
}

/// Key-less device on the first connection; encrypted with the received key afterwards.
async fn fresh_device() -> (String, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let (key_tx, key_rx) = oneshot::channel();
    tokio::spawn(async move {
        // 1. plaintext, no key
        let (mut s, _) = listener.accept().await.unwrap();
        let mut key = None;
        while let Ok(m) = read_msg(&mut s).await {
            match m {
                Msg::HelloRequest(_) => {
                    write_msg(
                        &mut s,
                        pb::HelloResponse {
                            api_version_major: 1,
                            api_version_minor: 12,
                            server_info: "fake".into(),
                            name: "home-assistant-voice-test".into(),
                        },
                    )
                    .await
                }
                Msg::DeviceInfoRequest(_) => {
                    write_msg(
                        &mut s,
                        pb::DeviceInfoResponse {
                            name: "home-assistant-voice-test".into(),
                            api_encryption_supported: true,
                            ..Default::default()
                        },
                    )
                    .await
                }
                Msg::NoiseEncryptionSetKeyRequest(r) => {
                    key = Some(String::from_utf8(r.key).unwrap());
                    write_msg(&mut s, pb::NoiseEncryptionSetKeyResponse { success: true }).await;
                    // like ESPHome: apply the key, kick every client
                    write_msg(&mut s, pb::DisconnectRequest::default()).await;
                    break;
                }
                _ => {}
            }
        }
        let key = key.expect("client never sent a key");
        key_tx.send(key.clone()).unwrap();

        // 2. from now on only the new key works (the client may retry while we switch)
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            let api = EspHomeApi::builder()
                .name("home-assistant-voice-test".to_string())
                .encryption_key(key.clone())
                .build()
                .unwrap();
            let Ok(conn) = api.start(stream).await else {
                continue;
            };
            let tx = conn.sender();
            let mut rx = conn.receiver();
            while let Ok(msg) = rx.recv().await {
                if let ProtoMessage::ListEntitiesRequest(_) = msg {
                    tx.send(ProtoMessage::ListEntitiesDoneResponse(
                        npb::ListEntitiesDoneResponse {},
                    ))
                    .await
                    .unwrap();
                }
            }
        }
    });
    (addr, key_rx)
}

#[tokio::test]
async fn pair_sets_a_fresh_key_and_reconnects_encrypted() {
    let (addr, device_key) = fresh_device().await;
    let (pe, key) = VoicePe::pair(&addr).await.unwrap();

    assert_eq!(device_key.await.unwrap(), key);
    use base64::Engine as _;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(&key)
        .unwrap();
    assert_eq!(raw.len(), 32);

    let info = pe.info().await.unwrap();
    assert_eq!(info.name, "home-assistant-voice-test");
}

#[tokio::test]
async fn pair_refuses_a_device_that_already_has_a_key() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let api = EspHomeApi::builder()
            .name("keyed".to_string())
            .encryption_key("px7tsbK3C7bpXHr2OevEV2ZMg/FrNBw2+O2pNPbedtA=".to_string())
            .build()
            .unwrap();
        let _ = api.start(stream).await;
    });
    let err = VoicePe::pair(&addr).await.err().expect("must fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("factory reset"), "unexpected error: {msg}");
}
