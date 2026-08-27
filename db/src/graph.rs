use std::collections::HashMap;

use crate::encoder::Encoder;
use crate::tokenizer::tokenize;
use crate::types::Record;

pub const KEYWORDS_PER_DOC: usize = 6;
const NEIGHBORS_PER_KW: usize = 3;
const MAX_EXPANDED_DOCS: usize = 64;

#[derive(Default)]
pub struct AssocGraph {
    post: HashMap<u64, Vec<u64>>,
    adj: HashMap<u64, HashMap<u64, f32>>,
}

pub fn top_keywords(text: &str, df: &HashMap<u64, u32>, ndocs: u32, k: usize) -> Vec<String> {
    let mut tf: HashMap<String, u32> = HashMap::new();
    for w in tokenize(text) {
        if w.chars().count() < 2 || w.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        *tf.entry(w).or_insert(0) += 1;
    }
    let mut scored: Vec<(String, f32)> = tf
        .into_iter()
        .map(|(w, c)| {
            let idf = Encoder::idf_of(Encoder::hash_str(&w), df, ndocs);
            (w, c as f32 * idf)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k);
    scored.into_iter().map(|(w, _)| w).collect()
}

impl AssocGraph {
    pub fn add_doc(&mut self, doc_id: u64, keywords: &[String]) {
        let hashes: Vec<u64> = keywords.iter().map(|w| Encoder::hash_str(w)).collect();
        for h in &hashes {
            let post = self.post.entry(*h).or_default();
            if !post.contains(&doc_id) {
                post.push(doc_id);
            }
        }
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                *self.adj.entry(hashes[i]).or_default().entry(hashes[j]).or_insert(0.0) += 1.0;
                *self.adj.entry(hashes[j]).or_default().entry(hashes[i]).or_insert(0.0) += 1.0;
            }
        }
    }

    pub fn remove_doc(&mut self, doc_id: u64, keywords: &[String]) {
        for kw in keywords {
            let h = Encoder::hash_str(kw);
            if let Some(v) = self.post.get_mut(&h) {
                v.retain(|&x| x != doc_id);
            }
        }
    }

    pub fn neighbors(&self, kw_hash: u64) -> Vec<(u64, f32)> {
        let mut v: Vec<(u64, f32)> = self
            .adj
            .get(&kw_hash)
            .map(|m| m.iter().map(|(&k, &w)| (k, w)).collect())
            .unwrap_or_default();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(NEIGHBORS_PER_KW);
        v
    }

    pub fn postings(&self, kw_hash: u64) -> &[u64] {
        self.post.get(&kw_hash).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn expand_candidates(
        &self,
        query_keywords: &[String],
        is_live: impl Fn(u64) -> bool,
    ) -> HashMap<u64, f32> {
        let mut out: HashMap<u64, f32> = HashMap::new();
        let matched: Vec<u64> = query_keywords
            .iter()
            .map(|w| Encoder::hash_str(w))
            .filter(|h| self.post.get(h).map_or(false, |v| !v.is_empty()))
            .collect();
        if matched.is_empty() {
            return out;
        }
        let max_post = self
            .post
            .values()
            .map(|v| v.len())
            .max()
            .unwrap_or(1)
            .max(1) as f32;
        for qh in &matched {
            for (nh, weight) in self.neighbors(*qh) {
                let aff = (weight / max_post).min(1.0);
                for &doc in self.postings(nh) {
                    if !is_live(doc) || out.contains_key(&doc) {
                        continue;
                    }
                    out.insert(doc, aff);
                    if out.len() >= MAX_EXPANDED_DOCS {
                        return out;
                    }
                }
            }
        }
        out
    }
}

pub fn graph_keywords_for(record: &Record, df: &HashMap<u64, u32>, ndocs: u32) -> Vec<String> {
    top_keywords(&record.text, df, ndocs, KEYWORDS_PER_DOC)
}

