use std::fmt::Write as _;
use std::time::Instant;

use engram_db::collection::RecallOpts;
use engram_db::db::Db;
use engram_db::RememberOpts;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn pick_str<'a>(&mut self, v: &[&'a str]) -> &'a str {
        v[(self.next() % v.len() as u64) as usize]
    }
    fn pick_topic<'a>(&mut self, v: &[&'a [&'a str]]) -> &'a [&'a str] {
        v[(self.next() % v.len() as u64) as usize]
    }
}

const TOPICS: [&[&str]; 6] = [
    &["postgres", "replication", "wal", "vacuum", "btree", "failover"],
    &["kubernetes", "pod", "rollout", "ingress", "cluster", "helm"],
    &["rust", "borrow", "lifetime", "cargo", "crates", "async"],
    &["redis", "cache", "eviction", "latency", "shard", "pipeline"],
    &["vector", "embedding", "recall", "quantization", "ann", "index"],
    &["meeting", "roadmap", "quarterly", "standup", "planning", "review"],
];

const FILLER: &[&str] = &[
    "team", "system", "update", "issue", "note", "detail", "session", "report",
];

fn make_doc(rng: &mut Rng) -> String {
    let t = rng.pick_topic(&TOPICS);
    let mut s = String::new();
    for _ in 0..3 {
        s.push_str(rng.pick_str(t));
        s.push(' ');
        s.push_str(rng.pick_str(FILLER));
        s.push(' ');
    }
    s.trim().to_string()
}

fn percentile(v: &mut Vec<u128>, p: f64) -> u128 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    let idx = ((v.len() as f64 - 1.0) * p).round() as usize;
    v[idx.min(v.len() - 1)]
}

fn main() {
    let dir = std::env::temp_dir().join("engram-eval");
    let _ = std::fs::remove_dir_all(&dir);
    println!("| docs | insert docs/s | p50 us | p99 us | hit-rate@budget300 |");
    println!("|---|---|---|---|---|");

    for n_docs in [2_000usize, 10_000, 50_000] {
        let _ = std::fs::remove_dir_all(&dir);
        let db = Db::open(&dir).expect("open");
        let col = db.create_collection("bench").unwrap();
        let mut rng = Rng(0x9E3779B97F4A7C15);

        let t0 = Instant::now();
        let now = engram_db::unix_now();
        let span = 30 * 86400;
        for i in 0..n_docs {
            let d = make_doc(&mut rng);
            let et = now - ((rng.next() % span as u64) as i64);
            col.remember(RememberOpts::new(d).importance(0.3).event_time(et))
                .unwrap();
            let _ = i;
        }
        let insert_elapsed = t0.elapsed().as_secs_f64();

        let mut latencies: Vec<u128> = Vec::new();
        let mut hits_correct = 0usize;
        let queries = 120;
        for q in 0..queries {
            let ti = q % TOPICS.len();
            let probe = TOPICS[ti]
                .iter()
                .take(4)
                .copied()
                .collect::<Vec<&str>>()
                .join(" ");
            let t = Instant::now();
            let hits = col
                .recall(RecallOpts::new(&probe).budget_tokens(300).k_max(16))
                .unwrap();
            latencies.push(t.elapsed().as_micros());
            if let Some(top) = hits.first() {
                let expected = TOPICS[ti][0];
                if top.text.contains(expected) {
                    hits_correct += 1;
                }
            }
        }

        let tc = Instant::now();
        db.checkpoint_all().unwrap();
        let ckpt_ms = tc.elapsed().as_millis();
        let p50 = percentile(&mut latencies, 0.5);
        let p99 = percentile(&mut latencies, 0.99);
        let rate = hits_correct as f64 / queries as f64 * 100.0;
        let mut line = String::new();
        let _ = write!(
            line,
            "| {} | {:.0} | {} | {} | {:.1}% |",
            n_docs,
            n_docs as f64 / insert_elapsed,
            p50,
            p99,
            rate
        );
        println!("{line}");
        eprintln!(
            "checkpoint: {ckpt_ms} ms, dir size: {} KB",
            fs_size_kb(&dir)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn fs_size_kb(p: &std::path::Path) -> u64 {
    fn walk(p: &std::path::Path, acc: &mut u64) {
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let md = match e.metadata() {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if md.is_dir() {
                    walk(&e.path(), acc);
                } else {
                    *acc += md.len();
                }
            }
        }
    }
    let mut total = 0;
    walk(p, &mut total);
    total / 1024
}

