//! Vamana / DiskANN graph ported from Go `vse/engine/index/vamana.go`:
//! single-layer graph, robust prune with alpha, greedy beam search width L.
//!
//! Streaming-build quality measures (learned the hard way):
//! - insert-time pruning uses the FINAL alpha (strict α=1 kills long-range edges)
//! - multi-entry search: medoid entry + last-inserted + sampled seeds
//! - batch build shuffles insertion order, then alpha-refine pass

use super::{Cand, FlatStore, MaxHeap, Metric, MinHeap};

pub struct Vamana {
    pub store: FlatStore,
    pub metric: Metric,
    pub r: usize,
    pub l: usize,
    pub alpha: f32,
    entry: i32,
    last: u32,
    pub links: Vec<Vec<u32>>,
    pub ids: Vec<u64>,
    norms: Vec<f32>,
    visited: std::cell::RefCell<super::VisitedStamps>,
    seeds: Vec<u32>,
    refine_passes: usize,
}

impl Vamana {
    pub fn new(dim: usize, r: usize, l: usize, alpha: f32, metric: Metric) -> Self {
        Vamana {
            store: FlatStore::new(dim),
            metric,
            r: r.max(2),
            l: l.max(1),
            alpha: alpha.clamp(1.0, 2.0),
            entry: -1,
            last: 0,
            links: Vec::new(),
            ids: Vec::new(),
            norms: Vec::new(),
            visited: std::cell::RefCell::new(super::VisitedStamps::new()),
            seeds: Vec::new(),
            refine_passes: 1,
        }
    }

    /// Number of alpha-refinement passes after bulk insert (2-3 improves
    /// recall on overlapping-cluster data).
    pub fn with_refine_passes(mut self, n: usize) -> Self {
        self.refine_passes = n.clamp(1, 4);
        self
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
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

    /// Greedy beam search from multiple entry candidates
    /// (DiskANN SearchFromCandidates pattern).
    ///
    /// Termination-correctness detail: only the CLOSEST entry seeds the result
    /// heap; other entries/seeds go to the exploration frontier only. If far
    /// entries polluted the result heap, the worst-distance stop rule would
    /// fire immediately and the search would never leave the starting basin.
    fn beam_search_multi(&self, q_row: &[f32], eps: &[u32], l: usize) -> Vec<Cand> {
        let mut vis = self.visited.borrow_mut();
        vis.begin(self.store.len());

        let mut frontier = MinHeap::default();
        let mut results: MaxHeap = Default::default();

        let mut evaluated: Vec<Cand> = Vec::with_capacity(eps.len() + self.seeds.len());
        let consider = |node: u32, ev: &mut Vec<Cand>, vis: &mut super::VisitedStamps| {
            if vis.try_visit(node as usize) {
                let d = self.metric.raw(q_row, self.store.row(node as usize));
                ev.push(Cand { local: node, dist: d });
            }
        };
        for &e in eps {
            consider(e, &mut evaluated, &mut vis);
        }
        for &s in &self.seeds {
            consider(s, &mut evaluated, &mut vis);
        }
        evaluated.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(best) = evaluated.first() {
            frontier.push(*best);
            results.push(*best);
        }
        for c in evaluated.iter().skip(1) {
            frontier.push(*c);
        }

        while let Some(c) = frontier.pop() {
            if results.len() >= l && c.dist > results.peek_worst().map_or(f32::INFINITY, |w| w.dist)
            {
                break;
            }
            for &nb in &self.links[c.local as usize] {
                if !vis.try_visit(nb as usize) {
                    continue;
                }
                let nd = self.metric.raw(q_row, self.store.row(nb as usize));
                if results.len() < l || nd < results.peek_worst().map_or(f32::INFINITY, |w| w.dist)
                {
                    frontier.push(Cand { local: nb, dist: nd });
                    results.push(Cand { local: nb, dist: nd });
                    if results.len() > l {
                        results.pop();
                    }
                }
            }
        }
        drop(vis);
        results.into_sorted_vec()
    }

    #[allow(dead_code)]
    fn beam_search(&self, q_row: &[f32], ep: u32, l: usize) -> Vec<Cand> {
        let mut eps: Vec<u32> = Vec::with_capacity(self.seeds.len() + 1);
        eps.push(ep);
        eps.extend_from_slice(&self.seeds);
        self.beam_search_multi(q_row, &eps, l)
    }

    /// Robust prune — exact Go semantics (vamana.go robustPrune):
    /// candidates sorted by d(p,c); keep c iff for ALL selected r: α·d(r,c) >= d(p,c);
    /// short-circuit when |candidates| <= R; top up from discarded if short.
    fn robust_prune(&self, point: u32, cands: &[u32], alpha: f32) -> Vec<u32> {
        if cands.len() <= self.r {
            let mut out: Vec<u32> = cands.to_vec();
            out.retain(|&c| c != point);
            return out;
        }
        let mut scored: Vec<(u32, f32)> = cands
            .iter()
            .filter(|&&c| c != point)
            .map(|&c| (c, self.dist(point, c)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut out: Vec<u32> = Vec::with_capacity(self.r);
        let mut discarded: Vec<(u32, f32)> = Vec::new();
        for (c, dc) in scored {
            if out.len() >= self.r {
                break;
            }
            let good = out.iter().all(|&r| self.dist(r, c) * alpha >= dc);
            if good {
                out.push(c);
            } else {
                discarded.push((c, dc));
            }
        }
        let mut di = 0usize;
        while out.len() < self.r && di < discarded.len() {
            out.push(discarded[di].0);
            di += 1;
        }
        out
    }

    pub fn insert(&mut self, vec: &[f32]) -> u32 {
        self.insert_id(self.store.len() as u64, vec)
    }

    pub fn insert_id(&mut self, gid: u64, vec: &[f32]) -> u32 {
        let id = self.store.len() as u32;
        self.store.push(vec);
        self.ids.push(gid);
        if self.metric == Metric::Cosine {
            self.norms.push(crate::simd::dot_f32(vec, vec).sqrt().max(1e-12));
        }
        self.links.push(Vec::new());

        if self.entry < 0 {
            self.entry = id as i32;
            self.last = id;
            return id;
        }

        let q = self.store.row(id as usize);
        let eps = [self.entry.max(0) as u32, self.last];
        let cands: Vec<u32> = self
            .beam_search_multi(q, &eps, self.l.max(self.r))
            .into_iter()
            .map(|c| c.local)
            .collect();

        let new_links = self.robust_prune(id, &cands, self.alpha);
        for &n in &new_links {
            self.links[id as usize].push(n);
            self.links[n as usize].push(id);
        }

        for &n in new_links.iter().take(self.r) {
            if n == id {
                continue;
            }
            if self.links[n as usize].len() > self.r {
                let mut pool = std::mem::take(&mut self.links[n as usize]);
                pool.push(n);
                let pruned = self.robust_prune(n, &pool, self.alpha);
                self.links[n as usize] = pruned;
            }
        }

        self.last = id;

        // periodic cheap re-centering of the entry point on huge streams
        if id > 0 && id % 4096 == 0 {
            self.set_entry_medoid();
        }
        id
    }

    /// Batch build: shuffle insertion order (sequential correlated inserts
    /// starve cross-cluster edges), then refine + medoid entry + seeds.
    pub fn build(&mut self, rows: impl Iterator<Item = Vec<f32>>) {
        let batch: Vec<(u64, Vec<f32>)> = rows.into_iter().enumerate().map(|(i, r)| (i as u64, r)).collect();
        self.build_with_ids(batch.into_iter())
    }

    /// Build from (external_id, vector) pairs in SHUFFLED order — external ids
    /// decouple search results from insertion order.
    pub fn build_with_ids(&mut self, rows: impl Iterator<Item = (u64, Vec<f32>)>) {
        let mut batch: Vec<(u64, Vec<f32>)> = rows.collect();
        if std::env::var("ENGRAM_NO_SHUFFLE").is_err() {
            let mut rs = 0x51ED_2701u64;
            for i in (1..batch.len()).rev() {
                rs = rs.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = ((rs >> 33) as usize) % (i + 1);
                batch.swap(i, j);
            }
        }
        for (gid, row) in &batch {
            self.insert_id(*gid, row);
        }
        for _ in 0..self.refine_passes {
            self.refine_alpha_pass();
        }
        self.set_entry_medoid();
        self.seed_random_entries(16);
    }

    fn seed_random_entries(&mut self, k: usize) {
        let n = self.store.len() as u32;
        if n == 0 {
            return;
        }
        let mut rs = 0x9E37_79B9_u64;
        self.seeds.clear();
        for _ in 0..k.min(n as usize) {
            rs = rs.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.seeds.push(((rs >> 33) as u32) % n);
        }
    }

    /// Re-select every node's neighbors from a fresh beam search using the
    /// final alpha, with symmetric edge maintenance.
    fn refine_alpha_pass(&mut self) {
        let n = self.store.len() as u32;
        let alpha = self.alpha;
        for i in 0..n {
            let q = self.store.row(i as usize).to_vec();
            let eps = [self.entry.max(0) as u32, if i > 0 { i - 1 } else { i }];
            let mut cand_set: Vec<u32> = self
                .beam_search_multi(&q, &eps, self.l.max(self.r))
                .into_iter()
                .map(|c| c.local)
                .collect();
            for &old in &self.links[i as usize] {
                if !cand_set.contains(&old) {
                    cand_set.push(old);
                }
            }
            let pruned = self.robust_prune(i, &cand_set, alpha);
            let old_links = std::mem::take(&mut self.links[i as usize]);
            for &removed in &old_links {
                if !pruned.contains(&removed) {
                    self.links[removed as usize].retain(|&x| x != i);
                }
            }
            for &kept in &pruned {
                if kept != i && !self.links[kept as usize].contains(&i) {
                    self.links[kept as usize].push(i);
                }
            }
            self.links[i as usize] = pruned;
        }

        for i in 0..n {
            if self.links[i as usize].len() > self.r {
                let mut pool = std::mem::take(&mut self.links[i as usize]);
                pool.push(i);
                self.links[i as usize] = self.robust_prune(i, &pool, alpha);
            }
        }
    }

    /// Go parity: entry point should be the medoid (nearest-to-mean).
    fn set_entry_medoid(&mut self) {
        let n = self.store.len();
        if n == 0 {
            return;
        }
        let dim = self.store.dim;
        let mut mean = vec![0f32; dim];
        for i in 0..n {
            for (d, v) in self.store.row(i).iter().enumerate() {
                mean[d] += v / n as f32;
            }
        }
        let mut best = 0usize;
        let mut bd = f32::INFINITY;
        for i in 0..n {
            let d = crate::simd::l2_sq(&mean, self.store.row(i));
            if d < bd {
                bd = d;
                best = i;
            }
        }
        self.entry = best as i32;
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        self.search_with_l(query, top_k, self.l.max(top_k))
    }

    pub fn search_with_l(&self, query: &[f32], top_k: usize, l: usize) -> Vec<(u64, f32)> {
        if self.is_empty() {
            return Vec::new();
        }
        let mut eps: Vec<u32> = Vec::with_capacity(self.seeds.len() + 1);
        eps.push(self.entry.max(0) as u32);
        eps.extend_from_slice(&self.seeds);
        self.beam_search_multi(query, &eps, l.max(top_k))
            .into_iter()
            .take(top_k)
            .map(|c| (self.ids[c.local as usize], c.dist))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vamana_recall_vs_brute_force() {
        let dim = 16usize;
        let n = 800usize;
        let clusters = 8usize;
        let mut rng = 777u64;
        let mut nxt = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f32 / (1u32 << 31) as f32
        };
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let c = i % clusters;
                (0..dim)
                    .map(|_d| c as f32 * 10.0 + nxt())
                    .collect::<Vec<f32>>()
            })
            .collect();

        let mut vam = Vamana::new(dim, 16, 32, 1.2, Metric::L2);
        vam.build(rows.iter().cloned());

        let mut hits = 0;
        for qi in 0..30 {
            let qrow: Vec<f32> = (0..dim).map(|d| (qi % clusters) as f32 * 10.0 + d as f32).collect();
            let top: Vec<usize> = vam.search(&qrow, 10).into_iter().map(|(l, _)| l as usize).collect();
            let mut brute: Vec<(usize, f32)> = (0..n)
                .map(|i| (i, crate::simd::l2_sq(&qrow, rows[i].as_slice())))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let brute: Vec<usize> = brute.into_iter().take(10).map(|(i, _)| i).collect();
            hits += top.iter().filter(|t| brute.contains(t)).count();
        }
        let recall = hits as f32 / 300.0;
        assert!(recall > 0.8, "vamana recall {recall}");
    }
}

