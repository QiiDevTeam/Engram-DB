use serde::{Deserialize, Serialize};

pub const WORDS: usize = crate::types::DIM / 32;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Sketch {
    pub words: Vec<u64>,
    pub nz: u32,
}

impl Sketch {
    pub fn zero() -> Self {
        Sketch {
            words: vec![0u64; WORDS],
            nz: 0,
        }
    }

    pub fn from_dense(dense: &[f32], keep: usize) -> Self {
        debug_assert_eq!(dense.len(), crate::types::DIM);
        let mut idx: Vec<usize> = (0..dense.len()).filter(|&i| dense[i] != 0.0).collect();
        idx.sort_by(|&a, &b| {
            dense[b]
                .abs()
                .partial_cmp(&dense[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(keep);
        let mut words = vec![0u64; WORDS];
        let mut nz = 0u32;
        for &i in &idx {
            let v = dense[i];
            let code = if v > 0.0 {
                1u64
            } else if v < 0.0 {
                2u64
            } else {
                continue;
            };
            words[i / 32] |= code << (2 * (i % 32));
            nz += 1;
        }
        Sketch { words, nz }
    }

    pub fn is_empty(&self) -> bool {
        self.nz == 0
    }

    pub fn dot(&self, other: &Sketch) -> f32 {
        if self.is_empty() || other.is_empty() || self.words.len() != other.words.len() {
            return 0.0;
        }
        let acc = crate::simd::ternary_dot_count(&self.words, &other.words) as f32;
        acc / ((self.nz as f32) * (other.nz as f32)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_from(pairs: &[(usize, f32)]) -> Vec<f32> {
        let mut d = vec![0.0f32; crate::types::DIM];
        for &(i, v) in pairs {
            d[i] = v;
        }
        d
    }

    #[test]
    fn self_similarity_is_one() {
        let s = Sketch::from_dense(&dense_from(&[(0, 0.5), (100, -0.7), (999, 0.2)]), 192);
        assert!((s.dot(&s) - 1.0).abs() < 1e-5, "{}", s.dot(&s));
    }

    #[test]
    fn orthogonal_is_zero() {
        let a = Sketch::from_dense(&dense_from(&[(0, 1.0)]), 192);
        let b = Sketch::from_dense(&dense_from(&[(500, 1.0)]), 192);
        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn opposite_sign_negative() {
        let a = Sketch::from_dense(&dense_from(&[(3, 1.0), (7, 1.0)]), 192);
        let b = Sketch::from_dense(&dense_from(&[(3, -1.0), (7, -1.0)]), 192);
        assert!((a.dot(&b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn keep_limit_respected() {
        let mut pairs = Vec::new();
        for i in 0..500 {
            pairs.push((i, 0.01));
        }
        let s = Sketch::from_dense(&dense_from(&pairs), 192);
        let nz: u32 = s
            .words
            .iter()
            .map(|w| count_nonzero_pairs(*w))
            .sum();
        assert_eq!(nz, 192);
    }

    fn count_nonzero_pairs(mut w: u64) -> u32 {
        let mut n = 0;
        while w != 0 {
            if w & 3 != 0 {
                n += 1;
            }
            w >>= 2;
        }
        n
    }
}

