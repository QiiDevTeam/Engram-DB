//! Vamana recall tuning: alpha x searchL matrix on the adversarial
//! overlap-cluster 100k dataset, metric = r@10-in-ref50.

use std::time::Instant;

use engram_db::index::vamana::Vamana;
use engram_db::index::Metric;
use engram_db::simd;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0 >> 11
    }
}

const DIM: usize = 64;

fn main() {
    let n = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000usize);
    let clusters = 20usize;

    println!("building dataset n={n}...");
    let mut r = Rng(0xC0FFEE);
    let rows: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let c = i % clusters;
            (0..DIM)
                .map(|_d| c as f32 * 8.0 + ((r.next() % 1000) as f32 / 1000.0))
                .collect()
        })
        .collect();

    let mut qr = Rng(7);
    let queries: Vec<Vec<f32>> = (0..30)
        .map(|_| {
            let c = (qr.next() % clusters as u64) as f32;
            (0..DIM)
                .map(|_d| c * 8.0 + ((qr.next() % 1000) as f32 / 1000.0))
                .collect()
        })
        .collect();

    println!("computing brute top-50 reference...");
    let truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let mut v: Vec<(usize, f32)> = rows
                .iter()
                .enumerate()
                .map(|(i, row)| (i, simd::l2_sq(q, row)))
                .collect();
            v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            v.into_iter().take(50).map(|(i, _)| i).collect()
        })
        .collect();

    for &(alpha, r_cap) in &[(1.2f32, 32usize), (1.6, 32), (1.2, 48)] {
        let t0 = Instant::now();
        let v = {
            let mut vv =
                Vamana::new(DIM, r_cap, 96, alpha, Metric::L2).with_refine_passes(2);
            vv.build(rows.iter().cloned());
            vv
        };
        let build_s = t0.elapsed().as_secs_f64();

        let degs: f64 = (0..v.links.len())
            .map(|i| v.links[i].len())
            .sum::<usize>() as f64
            / v.links.len() as f64;

        print!(
            "alpha={alpha} R={r_cap} (build {:.0}s, avg_deg={degs:.1}):",
            build_s
        );
        for l in [64usize, 128, 256, 512] {
            let mut hits = 0usize;
            let mut lat = Vec::new();
            for (qi, q) in queries.iter().enumerate() {
                let t = Instant::now();
                let top: Vec<usize> = v
                    .search_with_l(q, 10, l)
                    .into_iter()
                    .map(|(gid, _)| gid as usize)
                    .collect();
                lat.push(t.elapsed().as_micros());
                hits += top.iter().filter(|g| truth[qi].contains(g)).count();
            }
            lat.sort_unstable();
            let p50 = lat[lat.len() / 2];
            print!("  L={l}: {:.0}% @{p50}us", hits as f64 / 3.0);
        }
        println!();
    }
}

