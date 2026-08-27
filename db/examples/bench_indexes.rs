//! Cross-index performance suite (Go-vse parity): HNSW vs Vamana vs IVF-PQ
//! on dense f32, plus Collection-level full-scan vs graph hot path.

use std::fmt::Write as _;
use std::time::Instant;

use engram_db::collection::RecallOpts;
use engram_db::db::Db;
use engram_db::index::{hnsw::Hnsw, ivfpq::IvfPq, vamana::Vamana, Metric};
use engram_db::sketch_hnsw;
use engram_db::simd;
use engram_db::RememberOpts;
use std::sync::atomic::Ordering;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0 >> 11
    }
}

const DIM: usize = 64;

fn gen_cluster(n: usize, clusters: usize) -> Vec<Vec<f32>> {
    let mut r = Rng(0xC0FFEE);
    (0..n)
        .map(|i| {
            let c = i % clusters;
            (0..DIM)
                .map(|d| c as f32 * 8.0 + ((r.next() % 1000) as f32 / 1000.0) + d as f32 * 1e-4)
                .collect()
        })
        .collect()
}

fn brute_topk(rows: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut v: Vec<(usize, f32)> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (i, simd::l2_sq(q, r)))
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    v.into_iter().take(k).map(|(i, _)| i).collect()
}

fn percentile(lat_us: &mut [u128], p: f64) -> u128 {
    lat_us.sort_unstable();
    let idx = ((lat_us.len() as f64 - 1.0) * p).round() as usize;
    lat_us[idx.min(lat_us.len() - 1)]
}

type SearchFn<'a> = Box<dyn Fn(usize, &[f32], usize) -> Vec<usize> + 'a>;

fn bench_dense_index<'a>(
    name: &str,
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    build: impl FnOnce() -> SearchFn<'a>,
) -> String {
    let t0 = Instant::now();
    let search = build();
    let build_ms = t0.elapsed().as_millis();

    let mut lat = Vec::new();
    let mut hits = 0usize;
    let k = 10usize;
    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let top = search(qi, q, k);
        lat.push(t.elapsed().as_micros());
        for local in top {
            if truth[qi].contains(&(local as usize)) {
                hits += 1;
            }
        }
    }
    let recall = hits as f64 / (queries.len() * k) as f64;
    let p50 = percentile(&mut lat, 0.5);
    let p99 = percentile(&mut lat, 0.99);
    let mut out = String::new();
    let _ = write!(out, "| {name} | {build_ms} | {p50} | {p99} | {:.1}% |", recall * 100.0);
    out
}

fn main() {
    println!("== A. Collection memory engine: graph hot path vs full scan ==");
    collection_compare(50_000);

    println!("\n== B. Dense f32 suite (100k x {DIM}, L2, recall@10 vs AVX2 brute) ==");
    for geom in ["uniform", "overlap-cluster"] {
        let n = 100_000usize;
        let rows = match geom {
            "uniform" => gen_uniform(n),
            _ => gen_cluster(n, 20),
        };
        let mut rng = Rng(7);
        let queries: Vec<Vec<f32>> = (0..50)
            .map(|_| match geom {
                // queries follow the data distribution (realistic access pattern)
                "uniform" => (0..DIM)
                    .map(|_d| ((rng.next() % 100000) as f32 / 100000.0) * 160.0)
                    .collect(),
                _ => {
                    let c = (rng.next() % 20) as f32;
                    (0..DIM)
                        .map(|_d| c * 8.0 + ((rng.next() % 1000) as f32 / 1000.0))
                        .collect()
                }
            })
            .collect();

        println!("\n-- dataset: {geom} --");
        println!("| index | build ms | p50 us | p99 us | r@10-in-ref50 |");
        println!("|---|---|---|---|---|");
        {
            let mut lat = Vec::new();
            let mut hits = 0usize;
            let truth: Vec<Vec<usize>> =
                queries.iter().map(|q| brute_topk(&rows, q, 10)).collect();
            for (qi, q) in queries.iter().enumerate() {
                let t = Instant::now();
                let top = brute_topk(&rows, q, 10);
                lat.push(t.elapsed().as_micros());
                hits += top.iter().filter(|t| truth[qi].contains(t)).count();
            }
            println!(
                "| brute-force AVX2 | - | {} | {} | {:.1}% |",
                percentile(&mut lat, 0.5),
                percentile(&mut lat, 0.99),
                hits as f64 / (queries.len() * 10) as f64 * 100.0
            );
        }

        let line = bench_dense_index(
            "HNSW M=16 ef=128",
            &queries,
            &truth_of(&queries, &rows),
            || {
                let mut h = Hnsw::new(DIM, 16, 128, Metric::L2);
                for r in &rows {
                    h.insert(r);
                }
                Box::new(move |_qi, q, k| {
                    h.search(q, k).into_iter().map(|(l, _)| l as usize).collect()
                })
            },
        );
        println!("{line}");

        let line = bench_dense_index(
            "Vamana R=32 a=1.6 Lb=96 refx2 searchL=128",
            &queries,
            &truth_of(&queries, &rows),
            || {
                let mut v = Vamana::new(DIM, 32, 96, 1.6, Metric::L2).with_refine_passes(2);
                v.build(rows.iter().cloned());
                Box::new(move |_qi, q, k| {
                    v.search_with_l(q, k, 128)
                        .into_iter()
                        .map(|(l, _)| l as usize)
                        .collect()
                })
            },
        );
        println!("{line}");

        let line = bench_dense_index(
            "IVF-PQ nlist=1024 probe=32 m=16 resid rerank",
            &queries,
            &truth_of(&queries, &rows),
            || {
                let mut idx = IvfPq::new(DIM, 1024, 32, 16, Metric::L2);
                idx.train_and_build(rows.iter().cloned());
                Box::new(move |_qi, q, k| {
                    idx.search(q, k, true).into_iter().map(|(l, _)| l as usize).collect()
                })
            },
        );
        println!("{line}");
    }

    println!("\nGPU dll present: {}", engram_db::gpu::available());
}

fn gen_uniform(n: usize) -> Vec<Vec<f32>> {
    let mut r = Rng(0xF00D);
    (0..n)
        .map(|_| {
            (0..DIM)
                .map(|_d| ((r.next() % 100000) as f32 / 100000.0) * 160.0)
                .collect()
        })
        .collect()
}


fn collection_compare(n_docs: usize) {
    let dir = std::env::temp_dir().join("engram-bench-col");
    let _ = std::fs::remove_dir_all(&dir);
    let db = Db::open(&dir).unwrap();
    let col = db.create_collection("bench").unwrap();

    let topics = [
        "postgres replication wal vacuum failover",
        "kubernetes pod rollout ingress cluster",
        "rust borrow lifetime cargo async",
        "redis cache eviction shard latency",
        "vector embedding quantization ann index",
        "meeting roadmap quarterly standup review",
    ];
    let mut rng = Rng(42);
    let now = engram_db::unix_now();
    let t0 = Instant::now();
    for i in 0..n_docs {
        let t = topics[(rng.next() % topics.len() as u64) as usize];
        let et = now - ((rng.next() % (30 * 86400)) as i64);
        col.remember(RememberOpts::new(format!("{t} session {i} notes filler detail")).event_time(et))
            .unwrap();
    }
    let ins = t0.elapsed().as_secs_f64();
    println!("insert: {:.0} docs/s (graph auto-activated)", n_docs as f64 / ins);

    let probes: Vec<String> = topics
        .iter()
        .cycle()
        .take(60)
        .map(|t| t.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .collect();

    let mut lat_g = Vec::new();
    for p in &probes {
        let t = Instant::now();
        let hits = col.recall(RecallOpts::new(p).budget_tokens(400).k_max(16)).unwrap();
        lat_g.push(t.elapsed().as_micros());
        assert!(!hits.is_empty());
    }

    sketch_hnsw::GRAPH_THRESHOLD.store(usize::MAX, Ordering::Relaxed);
    let mut lat_s = Vec::new();
    for p in &probes {
        let t = Instant::now();
        let hits = col.recall(RecallOpts::new(p).budget_tokens(400).k_max(16)).unwrap();
        lat_s.push(t.elapsed().as_micros());
        assert!(!hits.is_empty());
    }
    sketch_hnsw::GRAPH_THRESHOLD.store(2048, Ordering::Relaxed);

    println!(
        "graph path : p50 {} us  p99 {} us",
        percentile(lat_g.clone().as_mut(), 0.5),
        percentile(lat_g.clone().as_mut(), 0.99)
    );
    println!(
        "full scan  : p50 {} us  p99 {} us",
        percentile(lat_s.clone().as_mut(), 0.5),
        percentile(lat_s.clone().as_mut(), 0.99)
    );
    let g50 = percentile(lat_g.as_mut(), 0.5) as f64;
    let s50 = percentile(lat_s.as_mut(), 0.5) as f64;
    println!("speedup: {:.1}x", s50 / g50.max(1.0));
}


fn truth_of(queries: &[Vec<f32>], rows: &[Vec<f32>]) -> Vec<Vec<usize>> {
    queries.iter().map(|q| brute_topk(rows, q, 50)).collect()
}



