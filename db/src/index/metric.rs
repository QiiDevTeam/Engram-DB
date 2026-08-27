/// Distance metrics backed by the SIMD layer (Go: MetricType + detectMetric).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    L2,
    Cosine,
    Dot,
}

impl Metric {
    #[inline]
    pub fn raw(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => crate::simd::l2_sq(a, b),
            Metric::Dot => crate::simd::dot_f32(a, b),
            Metric::Cosine => {
                let d = crate::simd::dot_f32(a, b);
                let na = crate::simd::dot_f32(a, a).sqrt();
                let nb = crate::simd::dot_f32(b, b).sqrt();
                1.0 - d / (na * nb).max(1e-12)
            }
        }
    }

    /// Monotone "larger = better" score for reranking outputs.
    #[inline]
    pub fn score_from_dist(&self, dist: f32) -> f32 {
        match self {
            Metric::L2 | Metric::Cosine => -dist,
            Metric::Dot => dist,
        }
    }
}

/// Precomputed norms for cosine over a FlatStore (Go: initMmapNorms).
pub struct Norms(pub Vec<f32>);

impl Norms {
    pub fn build(store: &crate::index::FlatStore) -> Norms {
        Norms((0..store.len())
            .map(|i| crate::simd::dot_f32(store.row(i), store.row(i)).sqrt().max(1e-12))
            .collect())
    }
}

