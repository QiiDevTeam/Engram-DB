//! SQ8 scalar quantizer (Go: NewSQQuantizer / Train / Encode / Decode).

use super::FlatStore;

pub struct Sq8 {
    pub dim: usize,
    pub min: Vec<f32>,
    pub scale: Vec<f32>,
}

impl Sq8 {
    pub fn train(store: &FlatStore) -> Sq8 {
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
        Sq8 { dim, min, scale }
    }

    pub fn encode(&self, row: &[f32]) -> Vec<u8> {
        row.iter()
            .zip(self.min.iter().zip(self.scale.iter()))
            .map(|(v, (&m, &s))| (((v - m) / s).round().clamp(0.0, 255.0)) as u8)
            .collect()
    }

    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        code.iter()
            .zip(self.min.iter().zip(self.scale.iter()))
            .map(|(&c, (&m, &s))| m + c as f32 * s)
            .collect()
    }

    pub fn l2_approx(&self, a: &[u8], b: &[u8], _min: &[f32], scale: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .enumerate()
            .map(|(d, (x, y))| {
                let dx = (*x as f32 - *y as f32) * scale[d];
                dx * dx
            })
            .sum::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq8_roundtrip() {
        let mut st = FlatStore::new(4);
        for i in 0..64 {
            st.push(&[(i % 10) as f32 * 3.1, i as f32 * 0.7, -(i as f32), 42.0]);
        }
        let q = Sq8::train(&st);
        for i in [0usize, 17, 63] {
            let row = st.row(i);
            let dec = q.decode(&q.encode(row));
            let err: f32 = row
                .iter()
                .zip(dec.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(err < 1.5, "row {i} err {err}");
        }
    }
}

