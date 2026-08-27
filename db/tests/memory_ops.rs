use engram_db::collection::{ConsolidateCfg, RecallOpts};
use engram_db::db::Db;
use engram_db::salience::Tier;
use engram_db::unix_now;
use engram_db::RememberOpts;

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "engram-m23-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn tiering_filters_cold_by_default() {
    let dir = tmpdir("tiers");
    let db = Db::open(&dir).unwrap();
    let col = db.create_collection("main").unwrap();

    let now = unix_now();
    for i in 0..12 {
        col.remember(
            RememberOpts::new(format!("ancient postgres incident report number {i} with postmortem details"))
                .importance(0.05)
                .event_time(1),
        )
        .unwrap();
    }
    col.remember(RememberOpts::new("fresh kubernetes deployment note today").event_time(now)).unwrap();

    let hits = col.recall(RecallOpts::new("postgres incident").budget_tokens(400)).unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits.iter().all(|h| h.tier != Tier::Cold),
        "cold docs must be filtered by default"
    );

    let cold_hits = col
        .recall(RecallOpts::new("postgres incident").include_cold(true).k_max(64).budget_tokens(2000))
        .unwrap();
    assert!(cold_hits.iter().any(|h| h.tier == Tier::Cold), "include_cold must admit cold tier");

    let st = col.detailed_stats();
    assert_eq!(st.hot + st.warm + st.cold, st.live);
    assert!(st.cold >= 12);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn consolidation_summarizes_and_archives() {
    let dir = tmpdir("consolidate");
    let db = Db::open(&dir).unwrap();
    let col = db.create_collection("main").unwrap();

    for i in 0..6 {
        col.remember(
            RememberOpts::new(format!("postgres tuning session {i}: adjusted work_mem and max_connections carefully"))
                .importance(0.05)
                .event_time(1),
        )
        .unwrap();
    }
    col.remember(
        RememberOpts::new("recent user preference: dark mode enabled everywhere")
            .importance(0.9)
            .event_time(unix_now()),
    )
    .unwrap();

    let report = col.consolidate(ConsolidateCfg::new()).unwrap();
    assert_eq!(report.clusters, 1, "{report:?}");
    assert_eq!(report.summaries_created, 1);
    assert_eq!(report.archived, 6);

    let st = col.detailed_stats();
    assert_eq!(st.summaries, 1);
    assert_eq!(st.archived, 6);

    let hits = col
        .recall(RecallOpts::new("postgres tuning work_mem").expand_summaries(true).include_cold(true))
        .unwrap();
    assert!(!hits.is_empty());
    let summary_hit = hits.iter().find(|h| h.text.starts_with("【摘要】"));
    assert!(summary_hit.is_some(), "summary must be recallable");
    let s = summary_hit.unwrap();
    assert!(!s.sources.is_empty(), "expanded sources must be present");
    assert!(s.sources.iter().any(|t| t.contains("work_mem")));

    db.checkpoint_all().unwrap();
    drop(col);
    drop(db);

    let db2 = Db::open(&dir).unwrap();
    let col2 = db2.collection("main").unwrap();
    let st2 = col2.detailed_stats();
    assert_eq!(st2.archived, 6, "archive must persist across reopen");
    assert_eq!(st2.summaries, 1);

    let fresh_hits = col2
        .recall(RecallOpts::new("dark mode preference"))
        .unwrap();
    assert!(!fresh_hits.is_empty());
    assert!(fresh_hits[0].text.contains("dark mode"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn association_graph_second_hop_recall() {
    let dir = tmpdir("graph");
    let db = Db::open(&dir).unwrap();
    let col = db.create_collection("main").unwrap();

    col.remember(RememberOpts::new("incident: postgres replication lag caused stale reads on replicas")).unwrap();
    col.remember(RememberOpts::new("runbook: postgres failover procedure requires promoting replica first")).unwrap();
    col.remember(
        RememberOpts::new(
            "architecture note: postgres and redis are colocated on the same node cluster",
        ),
    )
    .unwrap();
    col.remember(RememberOpts::new("redis cache eviction policy tuned for hot keys")).unwrap();
    col.remember(RememberOpts::new("gardening journal: tomatoes need more water in july")).unwrap();

    let direct = col.recall(RecallOpts::new("redis eviction policy")).unwrap();
    assert!(!direct.is_empty());

    let assoc = col
        .recall(
            RecallOpts::new("postgres failover promotion")
                .k_max(8)
                .budget_tokens(600)
                .include_cold(true),
        )
        .unwrap();
    let texts: Vec<&str> = assoc.iter().map(|h| h.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("replication lag")),
        "graph expansion should surface associated memory, got {texts:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

