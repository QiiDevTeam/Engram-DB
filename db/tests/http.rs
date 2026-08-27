use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

use engram_db::db::Db;
use serde_json::{json, Value};

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "engram-srv-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn http(port: u16, req: &str, body: Option<&Value>) -> (u16, Value) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    let msg = format!(
        "{req}\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    s.write_all(msg.as_bytes()).unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).unwrap();
    let text = String::from_utf8_lossy(&buf);
    let status: u16 = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body_start = text.find("\r\n\r\n").map(|i| i + 4).unwrap_or(text.len());
    let v: Value = serde_json::from_str(text[body_start..].trim()).unwrap_or(Value::Null);
    (status, v)
}

#[test]
fn server_end_to_end() {
    let dir = tmpdir("e2e");
    let db = Arc::new(Db::open(&dir).unwrap());
    let col = db.create_collection("main").unwrap();
    col.remember(
        engram_db::RememberOpts::new("user prefers dark mode and vim keybindings").importance(0.8),
    )
    .unwrap();
    drop(col);

    let addr = engram_db::server::spawn_server(db.clone()).unwrap();

    let (st, health) = http(addr.port(), "GET /health HTTP/1.1", None);
    assert_eq!(st, 200);
    assert_eq!(health["ok"], json!(true));

    let (st, r) = http(
        addr.port(),
        "POST /api/collections/main/remember HTTP/1.1",
        Some(&json!({"text": "deploy checklist always run migrations before rollout", "importance": 0.6})),
    );
    assert_eq!(st, 200, "{r}");
    assert!(r["id"].as_u64().is_some());

    let (st, r) = http(
        addr.port(),
        "POST /api/collections/main/recall HTTP/1.1",
        Some(&json!({"query": "what does the user prefer", "budget_tokens": 200})),
    );
    assert_eq!(st, 200, "{r}");
    let hits = r["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits[0]["text"].as_str().unwrap().contains("dark mode"),
        "{}",
        hits[0]
    );

    let (st, r) = http(addr.port(), "GET /api/collections/main/stats HTTP/1.1", None);
    assert_eq!(st, 200);
    assert_eq!(r["live"], json!(2));

    let (st, _) = http(addr.port(), "POST /api/checkpoint HTTP/1.1", None);
    assert_eq!(st, 200);

    // maintenance endpoints
    let (st, r) = http(
        addr.port(),
        "POST /api/checkpoint_keep_wal HTTP/1.1",
        None,
    );
    assert_eq!(st, 200, "{r}");

    let (st, r) = http(
        addr.port(),
        "POST /api/compact HTTP/1.1",
        Some(&json!({"retention_secs": 0})),
    );
    assert_eq!(st, 200, "{r}");

    let (st, r) = http(addr.port(), "GET /api/verify HTTP/1.1", None);
    assert_eq!(st, 200);
    assert_eq!(r["ok"], json!(true), "{r}");
    assert_eq!(r["rows"], json!(2));

    let (st, body) = {
        // raw export: ndjson bytes
        use std::io::Read;
        let mut s = std::net::TcpStream::connect(("127.0.0.1", addr.port())).unwrap();
        s.write_all(b"GET /api/export HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").unwrap();
        let mut buf = Vec::new();
        s.read_to_end(&mut buf).unwrap();
        let text = String::from_utf8_lossy(&buf).to_string();
        let status: u16 = text.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        (status, text)
    };
    assert_eq!(st, 200);
    assert!(body.contains("\"collection\":\"main\""));

    let (st, r) = http(
        addr.port(),
        "POST /api/backup HTTP/1.1",
        Some(&json!({"dest": dir.join("backup")})),
    );
    assert_eq!(st, 200, "{r}");
    assert!(dir.join("backup").join("collections").join("main").exists());


    let (st, r) = http(
        addr.port(),
        "POST /api/collections/bad!name/recall HTTP/1.1",
        Some(&json!({"query": "x"})),
    );
    assert_eq!(st, 400);

    // persistence check without re-opening the locked dir: export→import roundtrip
    let mut buf: Vec<u8> = Vec::new();
    {
        let cols = db.collection_names();
        assert_eq!(cols, vec!["main".to_string()]);
        let n = db.export_jsonl(&mut buf).unwrap();
        assert_eq!(n, 2);
    }

    let other = tmpdir("srv-other");
    let db2 = Db::open(&other).unwrap();
    let (imp, _) = db2.import_jsonl(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(imp, 2);
    let _ = std::fs::remove_dir_all(&dir);
}


