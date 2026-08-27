pub mod hnsw;
pub mod ivfpq;
pub mod metric;
pub mod sq;
pub mod vamana;

pub use metric::Metric;

/// Row-major dense vector store shared by all indexes (mirrors Go flatVecs).
#[derive(Clone, Default)]
pub struct FlatStore {
    pub dim: usize,
    pub data: Vec<f32>,
}

impl FlatStore {
    pub fn new(dim: usize) -> Self {
        FlatStore {
            dim,
            data: Vec::new(),
        }
    }

    pub fn with_capacity(dim: usize, n: usize) -> Self {
        FlatStore {
            dim,
            data: Vec::with_capacity(n * dim),
        }
    }

    pub fn push(&mut self, row: &[f32]) {
        debug_assert_eq!(row.len(), self.dim);
        self.data.extend_from_slice(row);
    }

    #[inline]
    pub fn len(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.data.len() / self.dim
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

/// Generation-stamped visited set — O(1) reset between searches
/// (Go: visitGen / visited []int32 pattern).
pub struct VisitedStamps {
    pub stamps: Vec<u32>,
    pub gen: u32,
}

impl VisitedStamps {
    pub fn new() -> Self {
        VisitedStamps {
            stamps: Vec::new(),
            gen: 0,
        }
    }

    #[inline]
    pub fn begin(&mut self, count: usize) {
        self.gen = self.gen.wrapping_add(1);
        if self.stamps.len() < count {
            self.stamps.resize(count, 0);
        }
        if self.gen == 0 {
            for s in self.stamps.iter_mut() {
                *s = u32::MAX;
            }
            self.gen = 1;
        }
    }

    #[inline]
    pub fn try_visit(&mut self, i: usize) -> bool {
        if self.stamps[i] == self.gen {
            false
        } else {
            self.stamps[i] = self.gen;
            true
        }
    }
}

impl Default for VisitedStamps {
    fn default() -> Self {
        Self::new()
    }
}

use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
pub struct Cand {
    pub local: u32,
    pub dist: f32,
}

/// Min-heap ordered by distance (closest on top) — candidate frontier.
#[derive(Default)]
pub struct MinHeap(pub BinaryHeap<std::cmp::Reverse<Cand>>);

impl MinHeap {
    pub fn push(&mut self, c: Cand) {
        self.0.push(std::cmp::Reverse(c));
    }
    pub fn pop(&mut self) -> Option<Cand> {
        self.0.pop().map(|r| r.0)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Max-heap ordered by distance (farthest on top) — bounded result set.
#[derive(Default)]
pub struct MaxHeap(pub BinaryHeap<Cand>);

impl MaxHeap {
    pub fn push(&mut self, c: Cand) {
        self.0.push(c);
    }
    pub fn pop(&mut self) -> Option<Cand> {
        self.0.pop()
    }
    pub fn peek_worst(&self) -> Option<&Cand> {
        self.0.peek()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn into_sorted_vec(self) -> Vec<Cand> {
        self.0.into_sorted_vec()
    }
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.local == other.local
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.local.cmp(&other.local))
    }
}

