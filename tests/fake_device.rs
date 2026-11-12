//! End-to-end test of our client against the device side of `esphome-native-api`,
//! acting as a fake Voice PE with a known Noise key.

use std::time::Duration;

use esphome_client::types as cl;
use esphome_native_api::esphomeapi::EspHomeApi;
use esphome_native_api::parser::ProtoMessage;
use esphome_native_api::proto as pb;
use havoice::api::Client;
use havoice::{Event, Rgb, VoicePe};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

const KEY: &str = "px7tsbK3C7bpXHr2OevEV2ZMg/FrNBw2+O2pNPbedtA=";

/// Starts a fake device; returns its address and a stream of commands it received.
async fn fake_device() -> (String, mpsc::Receiver<ProtoMessage>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let (seen_tx, seen_rx) = mpsc::channel(16);
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let api = EspHomeApi::builder()
            .name("home-assistant-voice-test".to_string())
            .encryption_key(KEY.to_string())
            .friendly_name("Fake Voice".to_string())
            .mac("20:F8:3B:00:00:01".to_string())
            .project_name("Nabu Casa.Home Assistant Voice PE".to_string())
            .project_version("26.9.0".to_string())
            .build()
            .unwrap();
        let Ok(conn) = api.start(stream).await else {
            return; // e.g. handshake failed on a wrong key
        };
        let tx = conn.sender();
        let mut rx = conn.receiver();
        while let Ok(msg) = rx.recv().await {
            if let ProtoMessage::ListEntitiesRequest(_) = msg {
                tx.send(ProtoMessage::ListEntitiesLightResponse(
                    pb::ListEntitiesLightResponse {
                        key: 42,
                        name: "LED Ring".into(),
                        effects: vec!["None".into(), "Rainbow".into()],
                        ..Default::default()
                    },
                ))
                .await
                .unwrap();
                tx.send(ProtoMessage::ListEntitiesEventResponse(
                    pb::ListEntitiesEventResponse {
                        key: 7,
                        name: "Button press".into(),
                        event_types: vec!["single_press".into(), "double_press".into()],
                        ..Default::default()
                    },
                ))
                .await
                .unwrap();
                tx.send(ProtoMessage::ListEntitiesDoneResponse(
                    pb::ListEntitiesDoneResponse {},
                ))
                .await
                .unwrap();
            } else if let ProtoMessage::SubscribeStatesRequest(_) = msg {
                // pretend someone double-pressed the centre button
                tx.send(ProtoMessage::EventResponse(pb::EventResponse {
                    key: 7,
                    event_type: "double_press".into(),
                    ..Default::default()
                }))
                .await
                .unwrap();
            } else {
                let _ = seen_tx.send(msg).await;
            }
        }
    });
    (addr, seen_rx)
}

#[tokio::test]
async fn encrypted_roundtrip() {
    let (addr, mut seen) = fake_device().await;
    let c = Client::connect(&addr, Some(KEY), "havoice-test")
        .await
        .unwrap();
    assert_eq!(c.hello.name, "home-assistant-voice-test");

    let info = c.device_info().await.unwrap();
    assert_eq!(info.friendly_name, "Fake Voice");
    assert_eq!(info.project_version, "26.9.0");

    // object_id is derived from the name when the device leaves it empty
    let ring = c.find("light", &["led_ring"]).await.unwrap();
    assert_eq!(ring.key, 42);
    assert_eq!(ring.choices, ["None", "Rainbow"]);

    c.send(cl::LightCommandRequest {
        key: ring.key,
        has_state: true,
        state: true,
        has_rgb: true,
        red: 1.0,
        green: 0.5,
        blue: 0.0,
        ..Default::default()
    })
    .await
    .unwrap();

    let got = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Some(ProtoMessage::LightCommandRequest(r)) = seen.recv().await {
                return r;
            }
        }
    })
    .await
    .expect("fake device never got the light command");
    assert_eq!(
        (got.key, got.state, got.red, got.green, got.blue),
        (42, true, 1.0, 0.5, 0.0)
    );
}

#[tokio::test]
async fn wrong_key_is_rejected() {
    let (addr, _seen) = fake_device().await;
    let err = Client::connect(
        &addr,
        Some("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        "t",
    )
    .await
    .err()
    .expect("connect should fail with a wrong key");
    let msg = format!("{err:#}");
    assert!(msg.contains("wrong HAVOICE_KEY"), "unexpected error: {msg}");
}

#[tokio::test]
async fn voice_pe_high_level() {
    let (addr, mut seen) = fake_device().await;
    let pe = VoicePe::connect(&addr, Some(KEY)).await.unwrap();
    assert_eq!(pe.led_effects(), ["None", "Rainbow"]);

    pe.led(Rgb(255, 0, 51), 1.0).await.unwrap();
    let got = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Some(ProtoMessage::LightCommandRequest(r)) = seen.recv().await {
                return r;
            }
        }
    })
    .await
    .unwrap();
    assert_eq!((got.key, got.red, got.green, got.blue), (42, 1.0, 0.0, 0.2));

    let mut events = pe.events().await.unwrap();
    let ev = tokio::time::timeout(Duration::from_secs(3), events.next())
        .await
        .unwrap();
    assert_eq!(ev, Some(Event::Button("double_press".into())));
}
