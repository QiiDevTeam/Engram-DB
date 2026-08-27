//! Isolated graph-quality debugger: prints degree stats + recall breakdown.

use engram_db::index::{hnsw::Hnsw, vamana::Vamana, Metric};

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
        .unwrap_or(5_000usize);
    let mode = std::env::args().nth(2).unwrap_or_else(|| "cluster".into());
    let clusters = 20usize;
    let mut r = Rng(0xC0FFEE);
    let rows: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            if mode == "random" {
                (0..DIM).map(|_d| ((r.next() % 100000) as f32 / 100000.0 * 160.0)).collect()
            } else {
                let c = i % clusters;
                (0..DIM)
                    .map(|_d| c as f32 * 8.0 + ((r.next() % 1000) as f32 / 1000.0))
                    .collect()
            }
        })
        .collect();
    let mut rng = Rng(7);
    let q: Vec<f32> = (0..DIM)
        .map(|_d| ((rng.next() % 20) as f32) * 8.0 + ((rng.next() % 1000) as f32 / 1000.0))
        .collect();

    // brute
    let mut bf: Vec<(usize, f32)> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| (i, engram_db::simd::l2_sq(&q, row)))
        .collect();
    bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let truth: Vec<usize> = bf.iter().take(10).map(|(i, _)| *i).collect();
    println!("brute dist range: {:.3} .. {:.3}", bf[0].1, bf[9].1);

    // vamana
    let mut v = Vamana::new(DIM, 16, 40, 1.2, Metric::L2);
    v.build(rows.iter().cloned());
    let degs: Vec<usize> = (0..n).map(|i| v.links[i].len()).collect();
    let empty = degs.iter().filter(|&&d| d == 0).count();
    let avg = degs.iter().sum::<usize>() as f64 / n as f64;
    println!("vamana: empty-link nodes={empty} avg_deg={avg:.1}");
    let top = v.search(&q, 10);
    let got: Vec<usize> = top.iter().map(|(l, _)| *l as usize).collect();
    println!("vamana top10: {got:?}");
    println!("truth      : {truth:?}");
    let hits = got.iter().filter(|g| truth.contains(g)).count();
    println!("vamana hits@10: {hits}");

    let degs: Vec<usize> = (0..n).map(|i| v.links[i].len()).collect();
    let empty = degs.iter().filter(|&&d| d == 0).count();
    let avg = degs.iter().sum::<usize>() as f64 / n as f64;
    let maxd = degs.iter().copied().max().unwrap_or(0);
    println!("vamana post-refine: empty={empty} avg={avg:.1} max={maxd}");

    // reachability from entry
    let entry = 0usize;
    let mut seen = vec![false; n];
    let mut stack = vec![entry];
    seen[entry] = true;
    let mut count = 1usize;
    while let Some(x) = stack.pop() {
        for &nb in &v.links[x] {
            if !seen[nb as usize] {
                seen[nb as usize] = true;
                count += 1;
                stack.push(nb as usize);
            }
        }
    }
    println!("reachable from node0: {count}/{n}");

    // multi-query recall sample with L sweep
    for l in [40usize, 100, 200, 400] {
        let mut total = 0usize;
        for qi in 0..20usize {
            let qc = (qi * 7 % 20) as f32;
            let mut qr = Rng(9000 + qi as u64);
            let qv: Vec<f32> = (0..DIM)
                .map(|_d| qc * 8.0 + ((qr.next() % 1000) as f32 / 1000.0))
                .collect();
            let mut b2: Vec<(usize, f32)> = rows
                .iter()
                .enumerate()
                .map(|(i, row)| (i, engram_db::simd::l2_sq(&qv, row)))
                .collect();
            b2.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let t2: Vec<usize> = b2.iter().take(10).map(|(i, _)| *i).collect();
            let got2: Vec<usize> = v
                .search_with_l(&qv, 10, l)
                .into_iter()
                .map(|(l2, _)| l2 as usize)
                .collect();
            total += got2.iter().filter(|g| t2.contains(g)).count();
        }
        println!("vamana L={l}: recall@10 {}/200", total);
    }

    // hnsw
    let mut h = Hnsw::new(DIM, 12, 96, Metric::L2);
    for row in &rows {
        h.insert(row);
    }
    let top_h = h.search(&q, 10);
    let got_h: Vec<usize> = top_h.iter().map(|(l, _)| *l as usize).collect();
    let hits_h = got_h.iter().filter(|g| truth.contains(g)).count();
    println!("hnsw hits@10: {hits_h}  top: {got_h:?}");

    // ivf-pq diagnostics
    use engram_db::index::ivfpq::IvfPq;
    use engram_db::index::Metric as M2;
    let mut idx = IvfPq::new(DIM, 316, 16, 8, M2::L2);
    idx.train_and_build(rows.iter().cloned());
    let mut sizes: Vec<usize> = idx.lists.iter().map(|l| l.len()).collect();
    sizes.sort_unstable();
    let empties = sizes.iter().filter(|&&s| s == 0).count();
    println!(
        "ivf lists: empties={} min={} med={} max={} total={}",
        empties,
        sizes.first().unwrap_or(&0),
        sizes[sizes.len() / 2],
        sizes.last().unwrap_or(&0),
        sizes.iter().sum::<usize>()
    );
    // which lists do the true neighbors live in?
    let cent = &idx.centroids;
    let list_of = |row: &[f32]| -> usize {
        let mut b = 0usize;
        let mut bd = f32::INFINITY;
        for (i, c) in cent.iter().enumerate() {
            let d = engram_db::simd::l2_sq(row, c);
            if d < bd {
                bd = d;
                b = i;
            }
        }
        b
    };
    let q_lists: Vec<usize> = {
        let mut v: Vec<(usize, f32)> = cent
            .iter()
            .enumerate()
            .map(|(i, c)| (i, engram_db::simd::l2_sq(&q, c)))
            .collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v.into_iter().take(8).map(|(i, _)| i).collect()
    };
    let truth_lists: Vec<usize> = truth.iter().map(|&i| list_of(rows[i].as_slice())).collect();
    println!("ivf: query's nearest-8 lists {q_lists:?}");
    println!("ivf: truth member lists {truth_lists:?}");
    let top_pq = idx.search(&q, 10, false);
    let top_rr = idx.search(&q, 10, true);
    println!(
        "ivfpq adc-only hits: {}  rerank hits: {}",
        top_pq.iter().filter(|(l, _)| truth.contains(&(*l as usize))).count(),
        top_rr.iter().filter(|(l, _)| truth.contains(&(*l as usize))).count()
    );
}

