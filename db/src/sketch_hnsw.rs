//! Specialized HNSW over ternary sketches — EngramDB collection hot path.
//! Distance = 1 - sign_cosine (monotone with match count), lazy deletion by
//! checking liveness at query time.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::sketch::Sketch;

/// Live-doc count above which the graph index activates.
pub static GRAPH_THRESHOLD: AtomicUsize = AtomicUsize::new(2048);

pub fn set_graph_threshold(n: usize) {
    GRAPH_THRESHOLD.store(n.max(16), Ordering::Relaxed);
}

pub struct SketchHnsw {
    pub m: usize,
    pub m0: usize,
    pub ef_search: usize,
    ml: f64,
    entry: i32,
    max_level: usize,
    sketches: Vec<Sketch>,
    ids: Vec<u64>,
    links: Vec<Vec<Vec<u32>>>,
    rng: u64,
    visited: std::cell::RefCell<crate::index::VisitedStamps>,
}

impl Default for SketchHnsw {
    fn default() -> Self {
        Self::new()
    }
}

impl SketchHnsw {
    pub fn new() -> Self {
        let m = 8usize;
        SketchHnsw {
            m,
            m0: 2 * m,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
            entry: -1,
            max_level: 0,
            sketches: Vec::new(),
            ids: Vec::new(),
            links: Vec::new(),
            rng: 0xA5A5_5A5A_5A5A_A5A5,
            visited: std::cell::RefCell::new(crate::index::VisitedStamps::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.sketches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sketches.is_empty()
    }

    fn next_level(&mut self) -> u8 {
        self.rng = self
            .rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = ((self.rng >> 11) as f64) / (1u64 << 53) as f64;
        ((-(r.ln()) * self.ml).exp() as u8).min(63)
    }

    #[inline]
    fn d_q(&self, q: &Sketch, i: u32) -> f32 {
        1.0 - q.dot(&self.sketches[i as usize])
    }

    #[inline]
    fn d(&self, a: u32, b: u32) -> f32 {
        1.0 - self.sketches[a as usize].dot(&self.sketches[b as usize])
    }

    pub fn insert(&mut self, gid: u64, sk: &Sketch) -> u32 {
        let local = self.sketches.len() as u32;
        self.sketches.push(sk.clone());
        self.ids.push(gid);
        let level = self.next_level() as usize;
        self.links.push(vec![Vec::new(); level + 1]);

        if self.entry < 0 {
            self.entry = local as i32;
            self.max_level = level;
            return local;
        }

        let mut ep = self.entry as u32;
        let mut ep_dist = self.d(local, ep);

        for lc in ((level + 1)..=self.max_level).rev() {
            loop {
                let mut improved = false;
                for &nb in &self.links[ep as usize][lc] {
                    let nd = self.d(local, nb);
                    if nd < ep_dist {
                        ep_dist = nd;
                        ep = nb;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
        }

        let mut ef = self.ef_search;
        for lc in (0..=level.min(self.max_level)).rev() {
            let cands = self.search_layer_local(local, ep, ep_dist, lc, ef);
            let cap = if lc == 0 { self.m0 } else { self.m };
            let selected = greedy_select(self, local, &cands, cap);
            if selected.is_empty() {
                break;
            }
            ep = selected[0].local;
            ep_dist = selected[0].dist;
            self.links[local as usize][lc] = selected.iter().map(|c| c.local).collect();
            for c in &selected {
                self.links[c.local as usize][lc].push(local);
                if self.links[c.local as usize][lc].len() > cap {
                    let mut pool: Vec<u32> = std::mem::take(&mut self.links[c.local as usize][lc]);
                    pool.push(c.local);
                    let pruned = prune_by_dist(self, c.local, &pool, cap);
                    self.links[c.local as usize][lc] = pruned;
                }
            }
            ef *= 2;
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = local as i32;
        }
        local
    }

    fn search_layer_local(
        &self,
        q: u32,
        ep: u32,
        ep_dist: f32,
        layer: usize,
        ef: usize,
    ) -> Vec<crate::index::Cand> {
        self.search_layer_sketch(&self.sketches[q as usize], ep, ep_dist, layer, ef)
    }

    fn search_layer_sketch(
        &self,
        q: &Sketch,
        ep: u32,
        ep_dist: f32,
        layer: usize,
        ef: usize,
    ) -> Vec<crate::index::Cand> {
        use crate::index::{Cand, MaxHeap, MinHeap};
        let mut vis = self.visited.borrow_mut();
        vis.begin(self.sketches.len());
        vis.try_visit(ep as usize);
        let mut frontier = MinHeap::default();
        let mut results: MaxHeap = Default::default();
        let start = Cand {
            local: ep,
            dist: ep_dist,
        };
        frontier.push(start);
        results.push(start);

        while let Some(c) = frontier.pop() {
            if results.len() >= ef && c.dist > results.peek_worst().map_or(f32::INFINITY, |w| w.dist)
            {
                break;
            }
            for &nb in &self.links[c.local as usize][layer] {
                if !vis.try_visit(nb as usize) {
                    continue;
                }
                let nd = self.d_q(q, nb);
                if results.len() < ef || nd < results.peek_worst().map_or(f32::INFINITY, |w| w.dist)
                {
                    frontier.push(Cand { local: nb, dist: nd });
                    results.push(Cand { local: nb, dist: nd });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        drop(vis);
        results.into_sorted_vec()
    }

    /// Returns global ids of approximate nearest neighbors, closest first.
    pub fn search(&self, q: &Sketch, top_k: usize, alive: impl Fn(u64) -> bool) -> Vec<(u64, f32)> {
        if self.is_empty() || self.entry < 0 {
            return Vec::new();
        }
        let ef = self.ef_search.max(top_k * 2);
        let mut ep = self.entry as u32;
        let mut ep_dist = self.d_q(q, ep);

        for lc in (1..=self.max_level).rev() {
            loop {
                let mut improved = false;
                for &nb in &self.links[ep as usize][lc] {
                    let nd = self.d_q(q, nb);
                    if nd < ep_dist {
                        ep_dist = nd;
                        ep = nb;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
        }

        self.search_layer_sketch(q, ep, ep_dist, 0, ef)
            .into_iter()
            .filter(|c| alive(self.ids[c.local as usize]))
            .take(top_k)
            .map(|c| (self.ids[c.local as usize], 1.0 - c.dist))
            .collect()
    }
}

fn greedy_select(me: &SketchHnsw, base: u32, cands: &[crate::index::Cand], cap: usize) -> Vec<crate::index::Cand> {
    let mut sorted = cands.to_vec();
    sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<crate::index::Cand> = Vec::with_capacity(cap);
    for c in &sorted {
        if out.len() >= cap {
            break;
        }
        if out.iter().all(|s| me.d(s.local, c.local) > c.dist) && c.local != base {
            out.push(*c);
        }
    }
    if out.is_empty() {
        for c in sorted.iter() {
            if c.local != base {
                out.push(*c);
                break;
            }
        }
    }
    out
}

fn prune_by_dist(me: &SketchHnsw, node: u32, pool: &[u32], cap: usize) -> Vec<u32> {
    let mut scored: Vec<crate::index::Cand> = pool
        .iter()
        .filter(|&&n| n != node)
        .map(|&n| crate::index::Cand {
            local: n,
            dist: me.d(node, n),
        })
        .collect();
    scored.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(cap);
    scored.into_iter().map(|c| c.local).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_sketch(seed: u64, band: usize) -> Sketch {
        let mut r = seed | 1;
        let mut dense = vec![0f32; crate::types::DIM];
        for _ in 0..192 {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = ((r >> 33) as usize) % crate::types::DIM;
            let v = if ((r >> 7) & 1) == 0 { band as f32 } else { -(band as f32) };
            dense[idx] += v;
        }
        Sketch::from_dense(&dense, 192)
    }

    #[test]
    fn sketch_hnsw_recall_agrees_with_brute() {
        let n = 600usize;
        let mut idx = SketchHnsw::new();
        let sketches: Vec<Sketch> = (0..n).map(|i| rand_sketch((i * 7919) as u64, i % 8)).collect();
        for (i, s) in sketches.iter().enumerate() {
            idx.insert(i as u64, s);
        }

        let mut hits = 0;
        let queries = 30usize;
        for qi in 0..queries {
            let q = rand_sketch(900_000 + qi as u64, qi % 8);
            let top: Vec<u64> = idx.search(&q, 10, |_| true).into_iter().map(|(id, _)| id).collect();

            let mut brute: Vec<(u64, f32)> = sketches
                .iter()
                .enumerate()
                .map(|(i, s)| (i as u64, q.dot(s)))
                .collect();
            brute.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let brute: Vec<u64> = brute.into_iter().take(10).map(|(i, _)| i).collect();

            hits += top.iter().filter(|t| brute.contains(t)).count();
        }
        let recall = hits as f32 / (queries * 10) as f32;
        assert!(recall > 0.7, "sketch hnsw recall {recall}");
    }
}

