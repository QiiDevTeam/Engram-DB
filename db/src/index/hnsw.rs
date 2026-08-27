//! HNSW ported from Go `vse/engine/index/flat_hnsw.go`:
//! multi-layer graph, greedy descent + ef-beam search, generation-stamped visited set.

use super::{Cand, FlatStore, MaxHeap, Metric, MinHeap};

pub struct Hnsw {
    pub store: FlatStore,
    pub metric: Metric,
    pub m: usize,
    pub m0: usize,
    pub ef_search: usize,
    pub ml: f64,
    entry: i32,
    max_level: usize,
    links: Vec<Vec<Vec<u32>>>,
    levels: Vec<u8>,
    norms: Vec<f32>,
    rng: u64,
    visited: std::cell::RefCell<super::VisitedStamps>,
}

impl Hnsw {
    pub fn new(dim: usize, m: usize, ef_search: usize, metric: Metric) -> Self {
        let m = m.max(2);
        Hnsw {
            store: FlatStore::new(dim),
            metric,
            m,
            m0: 2 * m,
            ef_search: ef_search.max(1),
            ml: 1.0 / (m as f64).ln(),
            entry: -1,
            max_level: 0,
            links: Vec::new(),
            levels: Vec::new(),
            norms: Vec::new(),
            rng: 0x9E37_79B9_7F4A_7C15,
            visited: std::cell::RefCell::new(super::VisitedStamps::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
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
    fn dist_q(&self, q_row: &[f32], i: u32) -> f32 {
        match self.metric {
            Metric::L2 => crate::simd::l2_sq(q_row, self.store.row(i as usize)),
            Metric::Dot => -crate::simd::dot_f32(q_row, self.store.row(i as usize)),
            Metric::Cosine => {
                1.0 - crate::simd::dot_f32(q_row, self.store.row(i as usize))
                    / self.norms[i as usize]
            }
        }
    }

    #[inline]
    fn dist(&self, a: u32, b: u32) -> f32 {
        match self.metric {
            Metric::L2 => crate::simd::l2_sq(self.store.row(a as usize), self.store.row(b as usize)),
            Metric::Dot => -crate::simd::dot_f32(self.store.row(a as usize), self.store.row(b as usize)),
            Metric::Cosine => {
                1.0 - crate::simd::dot_f32(self.store.row(a as usize), self.store.row(b as usize))
                    / (self.norms[a as usize] * self.norms[b as usize])
            }
        }
    }

    pub fn insert(&mut self, vec: &[f32]) -> u32 {
        let id = self.store.len() as u32;
        self.store.push(vec);
        if self.metric == Metric::Cosine {
            let n = crate::simd::dot_f32(vec, vec).sqrt().max(1e-12);
            self.norms.push(n);
        }
        let level = self.next_level() as usize;
        self.levels.push(level as u8);
        self.links.push(vec![Vec::new(); level + 1]);

        if self.entry < 0 {
            self.entry = id as i32;
            self.max_level = level;
            return id;
        }

        let mut ep = self.entry as u32;
        let mut ep_dist = self.dist(id, ep);

        for lc in ((level + 1)..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &nb in &self.links[ep as usize][lc] {
                    let d = self.dist(id, nb);
                    if d < ep_dist {
                        ep_dist = d;
                        ep = nb;
                        changed = true;
                    }
                }
            }
        }

        let ef = self.ef_search.max(self.m * 2);
        for lc in (0..=level.min(self.max_level)).rev() {
            let cands = self.search_layer(id, ep, ep_dist, lc, ef);
            let selected = self.select_neighbors_heuristic(id, &cands, if lc == 0 { self.m0 } else { self.m });
            ep = selected[0].local;
            ep_dist = selected[0].dist;
            for c in &selected {
                self.links[id as usize][lc].push(c.local);
                self.links[c.local as usize][lc].push(id);
                let cap = if lc == 0 { self.m0 } else { self.m };
                if self.links[c.local as usize][lc].len() > cap {
                    let pruned = self.prune_links(c.local, lc, cap);
                    self.links[c.local as usize][lc] = pruned;
                }
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = id as i32;
        }
        id
    }

    fn prune_links(&self, node: u32, lc: usize, cap: usize) -> Vec<u32> {
        let links = &self.links[node as usize][lc];
        let mut scored: Vec<Cand> = links
            .iter()
            .map(|&n| Cand {
                local: n,
                dist: self.dist(node, n),
            })
            .collect();
        scored.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(cap);
        scored.into_iter().map(|c| c.local).collect()
    }

    fn search_layer(&self, q: u32, ep: u32, ep_dist: f32, layer: usize, ef: usize) -> Vec<Cand> {
        self.search_layer_from(q_row_of(&self.store, q), ep, ep_dist, layer, ef)
    }

    fn search_layer_from(
        &self,
        q_row: &[f32],
        ep: u32,
        ep_dist: f32,
        layer: usize,
        ef: usize,
    ) -> Vec<Cand> {
        let mut vis = self.visited.borrow_mut();
        vis.begin(self.store.len());
        vis.try_visit(ep as usize);

        let mut frontier = MinHeap::default();
        frontier.push(Cand {
            local: ep,
            dist: ep_dist,
        });
        let mut results: MaxHeap = Default::default();
        results.push(Cand {
            local: ep,
            dist: ep_dist,
        });

        while let Some(c) = frontier.pop() {
            if results.len() >= ef && c.dist > results.peek_worst().map_or(f32::INFINITY, |w| w.dist) {
                break;
            }
            for &nb in &self.links[c.local as usize][layer] {
                if !vis.try_visit(nb as usize) {
                    continue;
                }
                let d = self.dist_q(q_row, nb);
                if results.len() < ef || d < results.peek_worst().map_or(f32::INFINITY, |w| w.dist) {
                    frontier.push(Cand { local: nb, dist: d });
                    results.push(Cand { local: nb, dist: d });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        drop(vis);
        results.into_sorted_vec()
    }

    /// Heuristic neighbor selection (HNSW alg.4) — keeps diverse edges.
    fn select_neighbors_heuristic(&self, base: u32, cands: &[Cand], cap: usize) -> Vec<Cand> {
        let mut sorted: Vec<Cand> = cands.to_vec();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        let mut out: Vec<Cand> = Vec::with_capacity(cap);
        for c in &sorted {
            if out.len() >= cap {
                break;
            }
            let ok = out
                .iter()
                .all(|s| self.dist(s.local, c.local) > c.dist);
            if ok {
                out.push(*c);
            }
        }
        if out.len() < cap {
            for c in &sorted {
                if out.len() >= cap {
                    break;
                }
                if !out.iter().any(|s| s.local == c.local) {
                    out.push(*c);
                }
            }
        }
        let _ = base;
        out
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(u32, f32)> {
        if self.is_empty() {
            return Vec::new();
        }
        let ef = self.ef_search.max(top_k);
        let mut ep = self.entry as u32;
        let mut ep_dist = self.dist_q(query, ep);

        for lc in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &nb in &self.links[ep as usize][lc] {
                    let d = self.dist_q(query, nb);
                    if d < ep_dist {
                        ep_dist = d;
                        ep = nb;
                        changed = true;
                    }
                }
            }
        }
        let cands = self.search_layer_from(query, ep, ep_dist, 0, ef);
        cands
            .into_iter()
            .take(top_k)
            .map(|c| (c.local, c.dist))
            .collect()
    }
}

fn q_row_of(store: &FlatStore, idx: u32) -> &[f32] {
    store.row(idx as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_clustered(n: usize, dim: usize, clusters: usize) -> (FlatStore, Vec<usize>) {
        let mut rng = 12345u64;
        let mut nxt = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f32 / (1u32 << 31) as f32
        };
        let mut st = FlatStore::with_capacity(dim, n);
        let mut truth = Vec::new();
        for i in 0..n {
            let c = i % clusters;
            let row: Vec<f32> = (0..dim)
                .map(|d| c as f32 * 10.0 + nxt() + d as f32 * 0.001)
                .collect();
            truth.push(c);
            st.push(&row);
        }
        (st, truth)
    }

    #[test]
    fn recall_vs_brute_force() {
        let (store, labels) = gen_clustered(1000, 32, 10);
        let mut hnsw = Hnsw::new(32, 8, 64, Metric::L2);
        for i in 0..store.len() {
            hnsw.insert(store.row(i));
        }

        let mut hits = 0usize;
        let queries = 50usize;
        for qi in 0..queries {
            let qrow: Vec<f32> = (0..32)
                .map(|d| (qi % 10) as f32 * 10.0 + d as f32 * 0.001)
                .collect();

            let graph_top: Vec<usize> = hnsw
                .search(&qrow, 10)
                .into_iter()
                .map(|(l, _)| l as usize)
                .collect();

            let mut brute: Vec<(usize, f32)> = (0..store.len())
                .map(|i| (i, crate::simd::l2_sq(&qrow, store.row(i))))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let brute: Vec<usize> = brute.into_iter().take(10).map(|(i, _)| i).collect();

            for g in &graph_top {
                if brute.contains(g) {
                    hits += 1;
                }
            }
            assert_eq!(labels[graph_top[0]], labels[brute[0]]);
        }
        let recall = hits as f32 / (queries * 10) as f32;
        assert!(recall > 0.85, "recall too low: {recall}");
    }
}

