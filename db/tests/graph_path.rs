use engram_db::collection::RecallOpts;
use engram_db::db::Db;
use engram_db::sketch_hnsw;
use engram_db::RememberOpts;
use std::sync::atomic::Ordering;
fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "engram-graph-{tag}-{}-{}",
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

#[test]
fn graph_path_agrees_with_full_scan() {
    let prev = sketch_hnsw::GRAPH_THRESHOLD.load(Ordering::Relaxed);
    sketch_hnsw::set_graph_threshold(64);

    let dir = tmpdir("agree");
    let db = Db::open(&dir).unwrap();
    let col = db.create_collection("main").unwrap();

    let topics = [
        "postgres replication lag failover wal",
        "kubernetes pod rollout ingress helm",
        "redis cache eviction shard pipeline",
        "rust borrow lifetime cargo async",
        "vector embedding quantization ann index",
        "meeting roadmap quarterly planning review",
    ];
    for i in 0..400 {
        let t = topics[i % topics.len()];
        col.remember(RememberOpts::new(format!("{t} note {i} detail filler word here"))).unwrap();
    }

    let probes = [
        "replication lag failover",
        "cache eviction shard",
        "cargo borrow lifetime",
        "quarterly roadmap planning",
    ];

    for p in probes {
        let graph_hits = col.recall(RecallOpts::new(p).budget_tokens(600).k_max(16)).unwrap();
        let scan_hits = col
            .recall(RecallOpts::new(p).budget_tokens(600).k_max(16).include_cold(true))
            .unwrap();
        assert!(!graph_hits.is_empty(), "{p}");
        assert!(!scan_hits.is_empty());

        let g_top_text = &graph_hits[0].text;
        assert!(
            topics.iter().any(|t| g_top_text.contains(t.split_whitespace().next().unwrap())),
            "graph top hit off-topic: {g_top_text}"
        );
        // top hit must contain the probe's leading keyword family
        let lead = p.split_whitespace().next().unwrap();
        assert!(
            g_top_text.contains(lead) || scan_hits[0].text.contains(lead),
            "probe '{p}': graph='{g_top_text}'"
        );

        // overlap between graph top8 and scan top8 should be meaningful
        let g8: Vec<u64> = graph_hits.iter().take(8).map(|h| h.id).collect();
        let s8: Vec<u64> = scan_hits.iter().take(8).map(|h| h.id).collect();
        let overlap = g8.iter().filter(|g| s8.contains(g)).count();
        assert!(overlap >= 3, "overlap {overlap} for probe '{p}': {g8:?} vs {s8:?}");
    }

    sketch_hnsw::set_graph_threshold(prev.max(16));
    let _ = std::fs::remove_dir_all(&dir);
}

