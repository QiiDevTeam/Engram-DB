use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SparseVec {
    pub idx: Vec<u32>,
    pub vals: Vec<f32>,
}

impl SparseVec {
    pub fn new(mut idx: Vec<u32>, mut vals: Vec<f32>) -> Self {
        assert_eq!(idx.len(), vals.len());
        let mut order: Vec<usize> = (0..idx.len()).collect();
        order.sort_by_key(|&i| idx[i]);
        idx = order.iter().map(|&i| idx[i]).collect();
        vals = order.iter().map(|&i| vals[i]).collect();
        SparseVec { idx, vals }
    }

    pub fn dot(&self, other: &SparseVec) -> f32 {
        let (mut i, mut j) = (0usize, 0usize);
        let mut acc = 0.0f32;
        while i < self.idx.len() && j < other.idx.len() {
            match self.idx[i].cmp(&other.idx[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    acc += self.vals[i] * other.vals[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        acc
    }

    pub fn is_empty(&self) -> bool {
        self.idx.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_dot_unordered_inputs() {
        let a = SparseVec::new(vec![9, 1, 5], vec![1.0, 2.0, 3.0]);
        let b = SparseVec::new(vec![5, 9], vec![3.0, 4.0]);
        assert!((a.dot(&b) - 13.0).abs() < 1e-6);
    }

    #[test]
    fn empty_vecs() {
        let a = SparseVec::default();
        let b = SparseVec::new(vec![1], vec![1.0]);
        assert_eq!(a.dot(&b), 0.0);
    }
}

