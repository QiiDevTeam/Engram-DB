use std::collections::HashMap;

use crate::sketch::Sketch;
use crate::sparse::SparseVec;
use crate::tokenizer::{char_trigrams, tokenize};

pub const TERNARY_KEEP: usize = 192;
pub const LEX_KEEP: usize = 128;

const W_WORD: f32 = 1.0;
const W_BIGRAM: f32 = 0.6;
const W_TRIGRAM: f32 = 0.45;

#[derive(Clone, Debug, Default)]
pub struct Encoder;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn hash_feature(feat: &str) -> u64 {
    splitmix64(feat.len() as u64 ^ 0x517C_C1B7_2722_0A95).wrapping_add({
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for b in feat.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100_0000_01B3);
        }
        h
    })
}

#[derive(Default)]
pub struct FeatureAcc {
    all: HashMap<u64, f32>,
    lex: HashMap<u64, f32>,
}

impl FeatureAcc {
    fn add(&mut self, feat: String, w: f32, lexical: bool) {
        let h = hash_feature(&feat);
        *self.all.entry(h).or_insert(0.0) += w;
        if lexical {
            *self.lex.entry(h).or_insert(0.0) += w;
        }
    }
}

pub fn collect_features(text: &str) -> FeatureAcc {
    let toks = tokenize(text);
    let mut acc = FeatureAcc::default();
    for (i, t) in toks.iter().enumerate() {
        acc.add(format!("w:{t}"), W_WORD, true);
        if i + 1 < toks.len() {
            acc.add(format!("b:{t}|{}", toks[i + 1]), W_BIGRAM, true);
        }
        for g in char_trigrams(t) {
            acc.add(format!("c:{g}"), W_TRIGRAM, false);
        }
    }
    acc
}

pub struct Encoded {
    pub sketch: Sketch,
    pub lex: SparseVec,
}

pub fn feature_hashes(text: &str) -> Vec<u64> {
    let acc = collect_features(text);
    let mut hs: Vec<u64> = acc.all.keys().copied().collect();
    hs.sort_unstable();
    hs
}

impl Encoder {
    pub fn new() -> Self {
        Encoder
    }

    pub fn hash_str(s: &str) -> u64 {
        hash_feature(s)
    }

    pub fn idf_of(h: u64, df: &HashMap<u64, u32>, ndocs: u32) -> f32 {
        Self::idf(df, ndocs, h)
    }

    fn idf(df: &HashMap<u64, u32>, ndocs: u32, h: u64) -> f32 {
        let d = df.get(&h).copied().unwrap_or(0) as f32;
        (((ndocs as f32 + 1.0) / (d + 1.0)).ln() + 1.0).max(0.05)
    }

    pub fn encode(
        &self,
        text: &str,
        df: &HashMap<u64, u32>,
        ndocs: u32,
    ) -> Encoded {
        let acc = collect_features(text);

        let mut dense = vec![0.0f32; crate::types::DIM];
        for (&h, &w) in &acc.all {
            let weight = w * Self::idf(df, ndocs, h);
            let bucket = (h % crate::types::DIM as u64) as usize;
            let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
            dense[bucket] += sign * weight;
        }
        let norm = dense.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in dense.iter_mut() {
                *v /= norm;
            }
        }
        let sketch = Sketch::from_dense(&dense, TERNARY_KEEP);

        let mut pairs: Vec<(u64, f32)> = acc
            .lex
            .iter()
            .map(|(&h, &w)| (h, w * Self::idf(df, ndocs, h)))
            .collect();
        pairs.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs.truncate(LEX_KEEP);
        let lnorm = pairs.iter().map(|(_, v)| v * v).sum::<f32>().sqrt();
        let mut idx = Vec::with_capacity(pairs.len());
        let mut vals = Vec::with_capacity(pairs.len());
        for (h, v) in pairs {
            if v == 0.0 {
                continue;
            }
            idx.push((h % u32::MAX as u64) as u32);
            vals.push(v / lnorm.max(1e-12));
        }
        let lex = SparseVec::new(idx, vals);

        Encoded { sketch, lex }
    }

    pub fn update_df(df: &mut HashMap<u64, u32>, text: &str, delta: i32) {
        for h in feature_hashes(text) {
            match delta {
                1 => *df.entry(h).or_insert(0) += 1,
                -1 => {
                    if let Some(v) = df.get_mut(&h) {
                        *v = v.saturating_sub(1);
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(text: &str) -> Encoded {
        let df = HashMap::new();
        Encoder::new().encode(text, &df, 0)
    }

    #[test]
    fn deterministic() {
        let a = enc("用户喜欢喝咖啡");
        let b = enc("用户喜欢喝咖啡");
        assert_eq!(a.sketch.words, b.sketch.words);
        assert_eq!(a.lex, b.lex);
    }

    #[test]
    fn related_beats_unrelated() {
        let q = enc("用户喜欢喝咖啡");
        let near = enc("用户每天喜欢喝茶和咖啡");
        let far = enc("quantum flux capacitor calibration");
        assert!(q.sketch.dot(&near.sketch) > q.sketch.dot(&far.sketch));
    }

    #[test]
    fn subword_similarity() {
        let a = enc("postgresql database migration failed");
        let b = enc("postgres database migration crashed");
        let c = enc("cherry blossom festival in kyoto");
        assert!(a.sketch.dot(&b.sketch) > a.sketch.dot(&c.sketch));
    }

    #[test]
    fn empty_text_gives_empty_channels() {
        let e = enc("");
        assert!(e.sketch.is_empty());
        assert!(e.lex.is_empty());
    }

    #[test]
    fn df_roundtrip() {
        let mut df = HashMap::new();
        let t = "alpha beta alpha";
        Encoder::update_df(&mut df, t, 1);
        assert!(!df.is_empty());
        Encoder::update_df(&mut df, t, -1);
        assert!(df.values().all(|&v| v == 0));
    }
}

