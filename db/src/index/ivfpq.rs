//! IVF-PQ ported from Go `vse/engine/quantizer/quantizer.go` + `index/ivf_pq.go`:
//! coarse k-means inverted lists + product-quantizer codes, ADC scan, nprobe, rerank.

use super::metric::Norms;
use super::{FlatStore, Metric};

/// Random (not strided!) sample of row indices — strided sampling over
/// cluster-ordered inserts silently drops whole clusters (gcd trap).
pub(crate) fn sample_row_indices(len: usize, target: usize) -> Vec<usize> {
    if len <= target {
        return (0..len).collect();
    }
    let mut idx: Vec<usize> = (0..len).collect();
    let mut rs = 0x2545_F491_4F6C_DD1Du64;
    for i in (1..idx.len()).rev() {
        rs = rs.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = ((rs >> 33) as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx.truncate(target);
    idx.sort_unstable();
    idx
}

pub struct PQ {
    pub sub_vecs: usize,
    pub sub_dim: usize,
    pub nbits: usize,
    pub centroids: Vec<Vec<Vec<f32>>>,
}

impl PQ {
    pub fn new(dim: usize, sub_vecs: usize) -> Self {
        let sub_vecs = sub_vecs.max(1).min(dim);
        let sub_dim = dim / sub_vecs;
        PQ {
            sub_vecs,
            sub_dim,
            nbits: 8,
            centroids: Vec::new(),
        }
    }

    pub fn trained(&self) -> bool {
        !self.centroids.is_empty()
    }

    /// Train PQ codebooks on RESIDUALS (x - nearest coarse centroid).
    pub fn train_residuals(
        &mut self,
        store: &FlatStore,
        centroids: &[Vec<f32>],
        iters: usize,
    ) {
        let keep = crate::index::ivfpq::sample_row_indices(store.len(), 10_000);
        let k = 1 << self.nbits;
        let mut cents = Vec::with_capacity(self.sub_vecs);
        for s in 0..self.sub_vecs {
            let off = s * self.sub_dim;
            let mut resid_rows: Vec<Vec<f32>> = Vec::with_capacity(keep.len());
            for &i in &keep {
                let row = store.row(i);
                let mut bd = f32::INFINITY;
                let mut bc = &centroids[0];
                for c in centroids {
                    let d = crate::simd::l2_sq(row, c);
                    if d < bd {
                        bd = d;
                        bc = c;
                    }
                }
                resid_rows.push(
                    row[off..off + self.sub_dim]
                        .iter()
                        .zip(bc[off..off + self.sub_dim].iter())
                        .map(|(a, b)| a - b)
                        .collect(),
                );
            }
            let refs: Vec<&[f32]> = resid_rows.iter().map(|r| r.as_slice()).collect();
            cents.push(self.kmeans_1d(&refs, k, iters));
        }
        self.centroids = cents;
    }

    pub fn encode_residual_row(&self, row: &[f32], coarse: &[f32]) -> Vec<u8> {
        let resid: Vec<f32> = row.iter().zip(coarse.iter()).map(|(a, b)| a - b).collect();
        self.encode_row(&resid)
    }

    /// ADC table against a list centroid: T[s][j] = Σ(-2·qc·Cr + Cr²),
    /// so that base(l2_sq(q,c)) + ΣT ≈ ||q - (c+r)||².
    pub fn adc_table_for(&self, q_minus_c: &[f32]) -> Vec<Vec<f32>> {
        let k = 1 << self.nbits;
        let mut table = Vec::with_capacity(self.sub_vecs);
        for s in 0..self.sub_vecs {
            let off = s * self.sub_dim;
            let qc = &q_minus_c[off..off + self.sub_dim];
            table.push(
                self.centroids[s]
                    .iter()
                    .take(k)
                    .map(|cr| {
                        let mut acc = 0f32;
                        for i in 0..self.sub_dim {
                            acc += -2.0 * qc[i] * cr[i] + cr[i] * cr[i];
                        }
                        acc
                    })
                    .collect(),
            );
        }
        table
    }

    fn kmeans_1d(&self, data: &[&[f32]], k: usize, iters: usize) -> Vec<Vec<f32>> {
        let dim = self.sub_dim;
        let mut rng = 0xDEAD_BEEF_CAFE_F00Du64;
        let mut next = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f32 / (1u32 << 31) as f32
        };
        if data.is_empty() {
            return vec![vec![0.0; dim.max(1)]; k];
        }
        // k-means++ lite: first centroid random, then D^2-weighted picks
        let mut cents: Vec<Vec<f32>> = Vec::with_capacity(k);
        cents.push(data[(next() * data.len() as f32) as usize % data.len()].to_vec());
        while cents.len() < k {
            let mut total = 0f64;
            let d2: Vec<f64> = data
                .iter()
                .map(|row| {
                    let best = cents
                        .iter()
                        .map(|c| crate::simd::l2_sq(row, c) as f64)
                        .fold(f64::INFINITY, f64::min);
                    total += best * best;
                    best * best
                })
                .collect();
            if total <= 1e-18 {
                cents.push(data[(next() * data.len() as f32) as usize % data.len()].to_vec());
                continue;
            }
            let mut target = next() as f64 / (1u64 << 31) as f64 * total;
            let mut pick = data.len() - 1;
            for (i, w) in d2.iter().enumerate() {
                target -= *w;
                if target <= 0.0 {
                    pick = i;
                    break;
                }
            }
            cents.push(data[pick].to_vec());
        }
        for _ in 0..iters {
            let mut sums = vec![vec![0f32; dim]; k];
            let mut counts = vec![0u32; k];
            for row in data {
                let mut best = 0usize;
                let mut bd = f32::INFINITY;
                for (ci, c) in cents.iter().enumerate() {
                    let d = crate::simd::l2_sq(row, c);
                    if d < bd {
                        bd = d;
                        best = ci;
                    }
                }
                counts[best] += 1;
                for d in 0..dim {
                    sums[best][d] += row[d];
                }
            }
            for ci in 0..k {
                if counts[ci] == 0 {
                    cents[ci] = data[(next() * data.len() as f32) as usize % data.len()].to_vec();
                } else {
                    for d in 0..dim {
                        cents[ci][d] = sums[ci][d] / counts[ci] as f32;
                    }
                }
            }
        }
        cents
    }

    pub fn train(&mut self, store: &FlatStore, iters: usize) {
        let k = 1 << self.nbits;
        let keep = crate::index::ivfpq::sample_row_indices(store.len(), 10_000);
        let rows: Vec<&[f32]> = keep.iter().map(|&i| store.row(i)).collect();
        let mut cents = Vec::with_capacity(self.sub_vecs);
        for s in 0..self.sub_vecs {
            let off = s * self.sub_dim;
            let sub_rows: Vec<&[f32]> = rows.iter().map(|r| &r[off..off + self.sub_dim]).collect();
            cents.push(self.kmeans_1d(&sub_rows, k, iters));
        }
        self.centroids = cents;
    }

    pub fn encode_row(&self, row: &[f32]) -> Vec<u8> {
        let mut code = vec![0u8; self.sub_vecs];
        for s in 0..self.sub_vecs {
            let off = s * self.sub_dim;
            let sub = &row[off..off + self.sub_dim];
            let mut best = 0u8;
            let mut bd = f32::INFINITY;
            for (ci, c) in self.centroids[s].iter().enumerate() {
                let d = crate::simd::l2_sq(sub, c);
                if d < bd {
                    bd = d;
                    best = ci as u8;
                }
            }
            code[s] = best;
        }
        code
    }

    /// ADC lookup table: [sub_vecs][256].
    pub fn adc_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let k = 1 << self.nbits;
        let mut table = Vec::with_capacity(self.sub_vecs);
        for s in 0..self.sub_vecs {
            let off = s * self.sub_dim;
            let sub = &query[off..off + self.sub_dim];
            table.push(
                self.centroids[s]
                    .iter()
                    .take(k)
                    .map(|c| crate::simd::l2_sq(sub, c))
                    .collect(),
            );
        }
        table
    }

    #[inline]
    pub fn distance_adc(&self, table: &[Vec<f32>], code: &[u8]) -> f32 {
        let mut acc = 0f32;
        for (s, &c) in code.iter().enumerate() {
            acc += table[s][c as usize];
        }
        acc
    }
}

pub struct SQ8 {
    pub min: Vec<f32>,
    pub scale: Vec<f32>,
}

impl SQ8 {
    pub fn train(store: &FlatStore) -> SQ8 {
        let dim = store.dim;
        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];
        for i in 0..store.len() {
            for (d, v) in store.row(i).iter().enumerate() {
                min[d] = min[d].min(*v);
                max[d] = max[d].max(*v);
            }
        }
        let scale: Vec<f32> = (0..dim)
            .map(|d| ((max[d] - min[d]) / 255.0).max(1e-12))
            .collect();
        SQ8 { min, scale }
    }

    pub fn encode(&self, row: &[f32]) -> Vec<u8> {
        row.iter()
            .zip(self.min.iter().zip(self.scale.iter()))
            .map(|(v, (&m, &s))| (((v - m) / s).round().clamp(0.0, 255.0)) as u8)
            .collect()
    }
}

pub struct IvfPq {
    pub store: FlatStore,
    pub metric: Metric,
    pub pq: PQ,
    pub centroids: Vec<Vec<f32>>,
    pub lists: Vec<Vec<u32>>,
    pub codes: Vec<Vec<u8>>,
    pub nprobe: usize,
    norms: Norms,
}

impl IvfPq {
    #[allow(unused_variables)]
    pub fn new(dim: usize, ncentroids: usize, nprobe: usize, sub_vecs: usize, metric: Metric) -> Self {
        IvfPq {
            store: FlatStore::new(dim),
            metric,
            pq: PQ::new(dim, sub_vecs),
            centroids: Vec::new(),
            lists: Vec::new(),
            codes: Vec::new(),
            nprobe: nprobe.max(1),
            norms: Norms(Vec::new()),
        }
    }

    pub fn train_and_build(&mut self, rows: impl Iterator<Item = Vec<f32>>) {
        let batch: Vec<Vec<f32>> = rows.collect();
        for r in &batch {
            self.store.push(r);
        }
        self.norms = Norms::build(&self.store);

        // IVF coarse k-means (randomly sampled, Go ivf.Train)
        let nlist = self.centroids_len_hint();
        let keep = crate::index::ivfpq::sample_row_indices(self.store.len(), 4096);
        let sample: Vec<&[f32]> = keep.iter().map(|&i| self.store.row(i)).collect();
        self.centroids = self.pq.kmeans_1d_full(&sample, nlist, 24);

        // assign to lists
        self.lists = vec![Vec::new(); self.centroids.len()];
        for i in 0..self.store.len() {
            let c = self.nearest_centroid(self.store.row(i));
            self.lists[c].push(i as u32);
        }

        // IVF-PQ proper: PQ codes encode RESIDUALS (x - assigned_centroid),
        // not raw vectors — slashes quantization error on structured data.
        self.pq.train_residuals(&self.store, &self.centroids, 10);
        let assign = |row: &[f32]| -> usize {
            let mut b = 0usize;
            let mut bd = f32::INFINITY;
            for (i, c) in self.centroids.iter().enumerate() {
                let d = crate::simd::l2_sq(row, c);
                if d < bd {
                    bd = d;
                    b = i;
                }
            }
            b
        };
        self.codes = (0..self.store.len())
            .map(|i| {
                let row = self.store.row(i);
                let c = &self.centroids[assign(row)];
                let resid: Vec<f32> =
                    row.iter().zip(c.iter()).map(|(a, b)| a - b).collect();
                self.pq.encode_row(&resid)
            })
            .collect();
    }

    fn centroids_len_hint(&self) -> usize {
        let n = self.store.len();
        ((n as f32).sqrt() as usize).clamp(4, 256)
    }

    fn nearest_centroid(&self, row: &[f32]) -> usize {
        let mut best = 0usize;
        let mut bd = f32::INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            let d = crate::simd::l2_sq(row, c);
            if d < bd {
                bd = d;
                best = i;
            }
        }
        best
    }

    /// ADC scan over probed lists with residual codes; exact rerank option.
    pub fn search(&self, query: &[f32], top_k: usize, rerank: bool) -> Vec<(u32, f32)> {
        if self.centroids.is_empty() || self.pq.centroids.is_empty() {
            return Vec::new();
        }
        let mut cdists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, crate::simd::l2_sq(query, c)))
            .collect();
        cdists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut scored: Vec<(u32, f32)> = Vec::new();
        for &(li, base) in cdists.iter().take(self.nprobe.min(self.lists.len())) {
            let c = &self.centroids[li];
            let q_minus_c: Vec<f32> =
                query.iter().zip(c.iter()).map(|(a, b)| a - b).collect();
            let table = self.pq.adc_table_for(&q_minus_c);
            for &local in &self.lists[li] {
                let d = base + self.pq.distance_adc(&table, &self.codes[local as usize]);
                scored.push((local, d));
            }
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate((top_k * 16).max(400));

        if rerank {
            for (local, d) in scored.iter_mut() {
                *d = self.metric.raw(query, self.store.row(*local as usize));
            }
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else if self.metric == Metric::Dot || self.metric == Metric::Cosine {
            for (_, d) in scored.iter_mut() {
                *d = -*d;
            }
        }
        scored.truncate(top_k);
        scored
    }
}

impl PQ {
    fn kmeans_1d_full(&self, data: &[&[f32]], k: usize, iters: usize) -> Vec<Vec<f32>> {
        // full-dim kmeans reused for IVF coarse centroids
        let dim = if data.is_empty() { 0 } else { data[0].len() };
        let fake = PQ {
            sub_vecs: 1,
            sub_dim: dim,
            nbits: 8,
            centroids: Vec::new(),
        };
        fake.kmeans_1d(data, k, iters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ivfpq_recall_with_rerank() {
        let dim = 16usize;
        let n = 2000usize;
        let clusters = 10usize;
        let mut rng = 4242u64;
        let mut nxt = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f32 / (1u32 << 31) as f32
        };
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let c = i % clusters;
                (0..dim).map(|d| c as f32 * 10.0 + nxt()).collect()
            })
            .collect();

        let mut idx = IvfPq::new(dim, 32, 4, 4, Metric::L2);
        idx.train_and_build(rows.clone().into_iter());

        let mut hits = 0;
        for qi in 0..20 {
            let qrow: Vec<f32> =
                (0..dim).map(|d| (qi % clusters) as f32 * 10.0 + d as f32).collect();
            let top: Vec<usize> = idx
                .search(&qrow, 10, true)
                .into_iter()
                .map(|(l, _)| l as usize)
                .collect();
            let mut brute: Vec<(usize, f32)> = (0..n)
                .map(|i| (i, crate::simd::l2_sq(&qrow, rows[i].as_slice())))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let brute: Vec<usize> = brute.into_iter().take(10).map(|(i, _)| i).collect();
            hits += top.iter().filter(|t| brute.contains(t)).count();
        }
        let recall = hits as f32 / 200.0;
        assert!(recall > 0.6, "ivfpq recall {recall}");
    }

    #[test]
    fn sq8_roundtrip_reasonable() {
        let dim = 8;
        let mut st = FlatStore::new(dim);
        for i in 0..100 {
            st.push(&(0..dim).map(|d| (i * d) as f32 % 50.0).collect::<Vec<f32>>());
        }
        let q = SQ8::train(&st);
        let row = st.row(42);
        let code = q.encode(row);
        let decoded: Vec<f32> = code
            .iter()
            .enumerate()
            .map(|(d, &c)| q.min[d] + c as f32 * q.scale[d])
            .collect();
        let err: f32 = row
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / dim as f32;
        assert!(err < 0.3, "sq8 mean abs err {err}");
    }
}

