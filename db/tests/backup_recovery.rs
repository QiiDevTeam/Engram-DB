use std::io::Write;

use engram_db::db::Db;
use engram_db::{Profile, RememberOpts};

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "engram-br-{tag}-{}-{}",
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

fn seed(db: &Db, col: &str, n: usize) {
    let c = db.create_collection(col).unwrap();
    for i in 0..n {
        c.remember(RememberOpts::new(format!("memory item {i} about rust and postgres")))
            .unwrap();
    }
}

#[test]
fn wal_torn_tail_recovers() {
    let dir = tmpdir("torn");

    let id_a;
    {
        let db = Db::open(&dir).unwrap();
        let col = db.create_collection("main").unwrap();
        id_a = col.remember(RememberOpts::new("first memory survives")).unwrap();
        drop(db);
    }

    // simulate crash mid-append: append a torn line to the live WAL
    let wal_path = dir.join("collections").join("main").join("wal.jsonl");
    let mut f = std::fs::OpenOptions::new().append(true).open(&wal_path).unwrap();
    f.write_all(b"{\"op\":\"remember\",\"row\":{\"record\":{\"id\":9").unwrap();
    f.sync_all().unwrap();
    drop(f);

    let db = Db::open(&dir).unwrap();
    let (live, _) = db.collection("main").unwrap().stats();
    assert_eq!(live, 1, "only the complete op survives a torn tail");
    let hits = db
        .collection("main")
        .unwrap()
        .recall(engram_db::collection::RecallOpts::new("first memory").budget_tokens(100))
        .unwrap();
    assert_eq!(hits[0].id, id_a);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cross_process_lock_blocks_second_open() {
    let dir = tmpdir("lock");
    let _db1 = Db::open(&dir).unwrap();

    match Db::open(&dir) {
        Err(engram_db::Error::Locked(_)) => {}
        other => panic!("expected Locked error, got {:?}", other.map(|_| "ok")),
    }

    drop(_db1);
    let _db2 = Db::open(&dir).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn backup_and_restore_roundtrip() {
    let dir = tmpdir("backup-src");
    let bak = tmpdir("backup-dst");
    let restore = tmpdir("restore");

    {
        let db = Db::open(&dir).unwrap();
        seed(&db, "main", 5);
        db.checkpoint_all().unwrap();
        db.backup_to(&bak).unwrap();
        // writes after the backup must NOT appear in the backup
        seed(&db, "main", 3);
    }

    assert!(bak.join("collections").join("main").join("manifest.json").exists());

    let restored = Db::restore_from(&bak, restore.join("data")).unwrap();
    let (_, total) = restored.collection("main").unwrap().stats();
    assert_eq!(total, 5, "post-backup writes must be absent after restore");

    let hits = restored
        .collection("main")
        .unwrap()
        .recall(engram_db::collection::RecallOpts::new("rust postgres").budget_tokens(200))
        .unwrap();
    assert!(!hits.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&bak);
    let _ = std::fs::remove_dir_all(&restore);
}

#[test]
fn export_import_roundtrip_preserves_ids() {
    let dir = tmpdir("export");
    let import_dir = tmpdir("import");

    let ids: Vec<u64>;
    {
        let db = Db::open(&dir).unwrap();
        let col = db.create_collection("notes").unwrap();
        ids = vec![
            col.remember(RememberOpts::new("alpha entry about kubernetes rollouts"))
                .unwrap(),
            col.remember(
                RememberOpts::new("beta summary of alpha").subject("note.beta"),
            )
            .unwrap(),
        ];
        db.checkpoint_all().unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let count = db.export_jsonl(&mut buf).unwrap();
        assert_eq!(count, 2);

        let db2 = Db::open(&import_dir).unwrap();
        let (imp, skip) = db2.import_jsonl(buf.as_slice()).unwrap();
        assert_eq!((imp, skip), (2, 0));

        // re-import is idempotent
        let (imp2, skip2) = db2.import_jsonl(std::io::Cursor::new(&buf)).unwrap();
        assert_eq!((imp2, skip2), (0, 2));

        for id in &ids {
            let hits = db2
                .collection("notes")
                .unwrap()
                .recall(
                    engram_db::collection::RecallOpts::new("kubernetes rollouts alpha")
                        .budget_tokens(300),
                )
                .unwrap();
            assert!(hits.iter().any(|h| h.id == *id), "id {id} lost in transit");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&import_dir);
}

#[test]
fn verify_detects_corruption() {
    let dir = tmpdir("verify");
    {
        let db = Db::open(&dir).unwrap();
        seed(&db, "main", 4);
        db.checkpoint_all().unwrap();

        let report = db.verify();
        assert!(report.ok(), "{:?}", report.errors);
        assert_eq!(report.rows, 4);

        // corrupt one row's importance via direct file surgery
        let seg = dir.join("collections").join("main").join("seg-000001.jsonl");
        let data = std::fs::read_to_string(&seg).unwrap();
        let patched = data.replace("\"importance\":0.5", "\"importance\":42.0");
        std::fs::write(&seg, patched).unwrap();
    }

    let reopened = Db::open(&dir).unwrap();
    let report = reopened.verify();
    assert!(!report.ok(), "corruption must be detected");
    assert!(report.errors.iter().any(|e| e.contains("importance")));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_drops_expired_but_keeps_summary_sources() {
    let dir = tmpdir("compact");
    let db = Db::open(&dir).unwrap();

    // Scenario 1: ACTIVE summary (live) protects its sources.
    {
        let col = db.create_collection("protected").unwrap();
        for i in 0..6 {
            col.remember(
                RememberOpts::new(format!("postgres tuning session {i} work_mem notes"))
                    .importance(0.05)
                    .event_time(1),
            )
            .unwrap();
        }
        col.consolidate(Default::default()).unwrap();
        assert_eq!(col.detailed_stats().archived, 6);

        let (a, b) = col.compact(86_400 * 30).unwrap(); // 30-day retention
        assert_eq!((a, b), (0, 0), "active summary protects its history");

        let hits = col
            .recall(
                engram_db::collection::RecallOpts::new("postgres tuning work_mem")
                    .expand_summaries(true),
            )
            .unwrap();
        let s = hits.iter().find(|h| h.text.starts_with("\u{3010}\u{6458}\u{8981}\u{3011}"));
        assert!(s.is_some());
        assert!(!s.unwrap().sources.is_empty());
    }

    // Scenario 2: forgetting the summary releases its sources for cleanup.
    {
        let col = db.create_collection("expired").unwrap();
        for i in 0..6 {
            col.remember(
                RememberOpts::new(format!("redis eviction notes {i}"))
                    .importance(0.05)
                    .event_time(1),
            )
            .unwrap();
        }
        col.consolidate(Default::default()).unwrap();

        // while the summary is live, history is protected
        let (a, b) = col.compact(0).unwrap();
        assert_eq!((a, b), (0, 0), "live summary protects its sources");

        for sid in col.summary_ids() {
            col.forget(sid).unwrap();
        }
        let (a, b) = col.compact(0).unwrap();
        assert_eq!((a, b), (0, 6), "forgotten summary releases its sources");
        assert_eq!(col.detailed_stats().archived, 0);
    }

    // Scenario 3: orphan archived rows (no summary at all) are removable.
    {
        let col = db.create_collection("orphans").unwrap();
        let row = engram_db::storage::SnapshotRow {
            record: engram_db::Record {
                id: 900,
                text: "orphan archived payload".into(),
                subject: None,
                tags: vec![],
                event_time: 1,
                ingest_time: 1,
                valid_to: None,
                importance: 0.1,
                hits: 0,
                last_hit: None,
                source_ids: vec![],
            },
            lex: Default::default(),
            sk: engram_db::sketch::Sketch::zero(),
            archived: true,
        };
        assert!(col.import_row(row).unwrap());
        let (a, b) = col.compact(0).unwrap();
        assert_eq!((a, b), (0, 1));
        assert_eq!(col.detailed_stats().archived, 0);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn checkpoint_keep_wal_enables_history_replay() {
    let dir = tmpdir("pitr");
    {
        let db = Db::open(&dir).unwrap();
        let col = db.create_collection("main").unwrap();
        col.remember(RememberOpts::new("pre-checkpoint memory A")).unwrap();
        db.checkpoint_all_keep_wal().unwrap();
        col.remember(RememberOpts::new("post-archive memory B")).unwrap();
    }

    let db = Db::open(&dir).unwrap();
    let (_, total) = db.collection("main").unwrap().stats();
    assert_eq!(total, 2, "archived WAL + live WAL both replay on reopen");

    let hits = db
        .collection("main")
        .unwrap()
        .recall(engram_db::collection::RecallOpts::new("memory B post archive").budget_tokens(150))
        .unwrap();
    assert!(hits.iter().any(|h| h.text.contains("memory B")));

    // profile query still routes correctly through Overview fallback etc.
    let _ = db
        .collection("main")
        .unwrap()
        .recall(engram_db::collection::RecallOpts::new("x").profile(Profile::Overview))
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

