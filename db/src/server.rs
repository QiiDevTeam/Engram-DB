use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use crate::collection::RecallOpts;
use crate::db::Db as CoreDb;
use crate::{Profile, RememberOpts};
use serde_json::{json, Value};

pub fn spawn_server(db: Arc<CoreDb>) -> std::io::Result<SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    thread::spawn(move || {
        for conn in listener.incoming() {
            match conn {
                Ok(stream) => {
                    let db = db.clone();
                    thread::spawn(move || {
                        let _ = handle_conn(&db, stream);
                    });
                }
                Err(_) => break,
            }
        }
    });
    Ok(addr)
}

pub fn serve_forever(db: CoreDb, addr: SocketAddr) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    eprintln!("engram-server listening on http://{addr}");
    let db = Arc::new(db);
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let db = db.clone();
                thread::spawn(move || {
                    let _ = handle_conn(&db, stream);
                });
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(reader: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_uppercase();
    let path = parts.next().unwrap_or("/").to_string();
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        reader.read_line(&mut h)?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Some(Request { method, path, body }))
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    body: &Value,
) -> std::io::Result<()> {
    let payload = serde_json::to_vec(body).unwrap_or_default();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    )?;
    stream.write_all(&payload)?;
    stream.flush()
}

fn handle_conn(db: &CoreDb, stream: TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut stream = stream;
    while let Some(req) = read_request(&mut reader)? {
        let body: Value =
            serde_json::from_slice(&req.body).unwrap_or(Value::Object(Default::default()));
        match route(db, &req.method, &req.path, body, &req.body) {
            RouteResp::Json(resp) => write_response(&mut stream, resp.0, &resp.1)?,
            RouteResp::Raw(status, ctype, payload) => {
                write_raw_response(&mut stream, status, &ctype, &payload)?
            }
        }
        return Ok(());
    }
    Ok(())
}

fn write_raw_response(
    stream: &mut TcpStream,
    status: u16,
    ctype: &str,
    payload: &[u8],
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    )?;
    stream.write_all(payload)?;
    stream.flush()
}

enum RouteResp {
    Json((u16, Value)),
    Raw(u16, String, Vec<u8>),
}

type Resp = (u16, Value);

fn route(db: &CoreDb, method: &str, path: &str, body: Value, raw_body: &[u8]) -> RouteResp {
    if method == "GET" && path == "/api/export" {
        let mut buf: Vec<u8> = Vec::new();
        return match db.export_jsonl(&mut buf) {
            Ok(_) => RouteResp::Raw(200, "application/x-ndjson".into(), buf),
            Err(e) => RouteResp::Json(err(500, e)),
        };
    }
    RouteResp::Json(route_json(db, method, path, body, raw_body))
}

fn route_json(db: &CoreDb, method: &str, path: &str, body: Value, raw_body: &[u8]) -> Resp {
    match (method, path) {
        ("GET", "/health") => (200, json!({"ok": true})),
        ("POST", "/api/checkpoint") => match db.checkpoint_all() {
            Ok(()) => (200, json!({"ok": true})),
            Err(e) => err(500, e),
        },
        ("POST", "/api/checkpoint_keep_wal") => match db.checkpoint_all_keep_wal() {
            Ok(()) => (200, json!({"ok": true})),
            Err(e) => err(500, e),
        },
        ("POST", "/api/compact") => {
            let retention = body.get("retention_secs").and_then(|v| v.as_u64()).unwrap_or(0);
            match db.compact(retention) {
                Ok((a, b)) => (
                    200,
                    json!({"live_dead_removed": a, "archived_removed": b}),
                ),
                Err(e) => err(500, e),
            }
        }
        ("GET", "/api/verify") => {
            let r = db.verify();
            (
                200,
                json!({
                    "collections": r.collections,
                    "rows": r.rows,
                    "ok": r.ok(),
                    "errors": r.errors,
                }),
            )
        }
        ("POST", "/api/import") => match db.import_jsonl(raw_body) {
            Ok((i, s)) => (200, json!({"imported": i, "skipped": s})),
            Err(e) => err(400, e),
        },
        ("POST", "/api/backup") => {
            let Some(dest) = body.get("dest").and_then(|v| v.as_str()).map(String::from) else {
                return err(400, "missing field: dest");
            };
            match db.backup_to(&dest) {
                Ok(p) => (200, json!({"ok": true, "path": p.to_string_lossy()})),
                Err(e) => err(500, e),
            }
        }
        _ => {
            if let Some(rest) = path.strip_prefix("/api/collections/") {
                let mut segs = rest.splitn(2, '/');
                let name = segs.next().unwrap_or("");
                let action = segs.next().unwrap_or("");
                return collection_route(db, method, name, action, body);
            }
            err(404, format!("no route: {method} {path}"))
        }
    }
}

fn collection_route(db: &CoreDb, method: &str, name: &str, action: &str, body: Value) -> Resp {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return err(400, "invalid collection name");
    }
    let col = match db.create_collection(name) {
        Ok(c) => c,
        Err(e) => return err(500, e),
    };
    match (method, action) {
        ("POST", "remember") => {
            let Some(text) = body.get("text").and_then(|v| v.as_str()).map(String::from) else {
                return err(400, "missing field: text");
            };
            let mut opts = RememberOpts::new(text).importance(
                body.get("importance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32,
            );
            if let Some(s) = body.get("subject").and_then(|v| v.as_str()) {
                opts = opts.subject(s.to_string());
            }
            if let Some(t) = body.get("event_time").and_then(|v| v.as_i64()) {
                opts = opts.event_time(t);
            }
            match col.remember(opts) {
                Ok(id) => (200, json!({"id": id})),
                Err(e) => err(500, e),
            }
        }
        ("POST", "recall") => {
            let Some(query) = body.get("query").and_then(|v| v.as_str()).map(String::from)
            else {
                return err(400, "missing field: query");
            };
            let profile = match body.get("profile").and_then(|v| v.as_str()).unwrap_or("chat") {
                "agent" | "agent-task" => Profile::AgentTask,
                "overview" => Profile::Overview,
                _ => Profile::Chat,
            };
            let opts = RecallOpts::new(query)
                .budget_tokens(body.get("budget_tokens").and_then(|v| v.as_u64()).unwrap_or(512) as usize)
                .k_max(body.get("k_max").and_then(|v| v.as_u64()).unwrap_or(64) as usize)
                .profile(profile);
            match col.recall(opts) {
                Ok(hits) => (
                    200,
                    json!({
                        "hits": hits.iter().map(|h| json!({
                            "id": h.id,
                            "score": h.score,
                            "text": h.text,
                            "estimated_tokens": h.estimated_tokens,
                            "tier": format!("{:?}", h.tier),
                            "sources": h.sources,
                        })).collect::<Vec<_>>()
                    }),
                ),
                Err(e) => err(500, e),
            }
        }
        ("GET", "stats") => {
            let st = col.detailed_stats();
            (
                200,
                json!({
                    "live": st.live,
                    "total_incl_dead": st.total_incl_dead,
                    "archived": st.archived,
                    "summaries": st.summaries,
                    "hot": st.hot,
                    "warm": st.warm,
                    "cold": st.cold,
                }),
            )
        }
        ("POST", "checkpoint") => match col.checkpoint() {
            Ok(()) => (200, json!({"ok": true})),
            Err(e) => err(500, e),
        },
        _ => err(404, format!("no collection action: {method} {action}")),
    }
}

fn err(code: u16, msg: impl std::fmt::Display) -> Resp {
    (code, json!({"error": msg.to_string()}))
}


