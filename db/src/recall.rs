#[derive(Clone, Copy, Debug)]
pub struct Weights {
    pub semantic: f32,
    pub lexical: f32,
    pub recency: f32,
    pub importance: f32,
    pub diversity: f32,
    pub graph: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Chat,
    AgentTask,
    Overview,
}

impl Profile {
    pub fn weights(self) -> Weights {
        match self {
            Profile::Chat => Weights {
                semantic: 0.5,
                lexical: 0.22,
                recency: 0.18,
                importance: 0.05,
                diversity: 0.35,
                graph: 0.05,
            },
            Profile::AgentTask => Weights {
                semantic: 0.55,
                lexical: 0.28,
                recency: 0.05,
                importance: 0.08,
                diversity: 0.25,
                graph: 0.04,
            },
            Profile::Overview => Weights {
                semantic: 0.6,
                lexical: 0.1,
                recency: 0.13,
                importance: 0.13,
                diversity: 0.45,
                graph: 0.04,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cand {
    pub doc_idx: usize,
    pub base: f32,
    pub text_len_chars: usize,
}

pub fn fuse(
    w: &Weights,
    semantic: f32,
    lexical: f32,
    recency: f32,
    importance: f32,
    graph_affinity: f32,
) -> f32 {
    w.semantic * semantic
        + w.lexical * lexical
        + w.recency * recency
        + w.importance * importance
        + w.graph * graph_affinity
}

pub fn mmr_select(
    mut cands: Vec<Cand>,
    budget_tokens: usize,
    k_max: usize,
    diversity: f32,
    doc_sim: &dyn Fn(usize, usize) -> f32,
) -> Vec<usize> {
    cands.sort_by(|a, b| {
        b.base
            .partial_cmp(&a.base)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut selected: Vec<Cand> = Vec::new();
    let mut used = 0usize;
    let mut taken = vec![false; cands.len()];
    let mut pens = vec![0f32; cands.len()];

    for _ in 0..k_max {
        let mut best_i: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;
        for (i, c) in cands.iter().enumerate() {
            if taken[i] || used + estimate_tokens_by(c.text_len_chars) > budget_tokens {
                continue;
            }
            let score = c.base - diversity * pens[i];
            if score > best_score {
                best_score = score;
                best_i = Some(i);
            }
        }
        let Some(i) = best_i else { break };
        used += estimate_tokens_by(cands[i].text_len_chars);
        taken[i] = true;
        let new_sel_doc = cands[i].doc_idx;
        for (j, c) in cands.iter().enumerate() {
            if taken[j] || pens[j] >= 1.0 {
                continue;
            }
            let s = doc_sim(c.doc_idx, new_sel_doc);
            if s > pens[j] {
                pens[j] = s;
            }
        }
        selected.push(Cand {
            doc_idx: new_sel_doc,
            base: cands[i].base,
            text_len_chars: cands[i].text_len_chars,
        });
    }

    selected.into_iter().map(|c| c.doc_idx).collect()
}

fn estimate_tokens_by(chars: usize) -> usize {
    (chars + crate::types::TOKEN_CHARS_PER_TOKEN - 1) / crate::types::TOKEN_CHARS_PER_TOKEN
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sim_map(pairs: &[(usize, usize, f32)]) -> impl Fn(usize, usize) -> f32 + '_ {
        let m: HashMap<(usize, usize), f32> = pairs
            .iter()
            .flat_map(|&(a, b, s)| [(a, b, s), (b, a, s)])
            .map(|(a, b, s)| ((a, b), s))
            .collect();
        move |a, b| *m.get(&(a, b)).unwrap_or(&0.0)
    }

    #[test]
    fn respects_budget() {
        let cands: Vec<Cand> = (0..10)
            .map(|i| Cand {
                doc_idx: i,
                base: 1.0 - i as f32 * 0.01,
                text_len_chars: 100,
            })
            .collect();
        let pick = mmr_select(cands.clone(), 120, 64, 0.0, &|_, _| 0.0);
        let used: usize = pick
            .iter()
            .map(|&i| estimate_tokens_by(cands[i].text_len_chars))
            .sum();
        assert!(used <= 120 && !pick.is_empty());
    }

    #[test]
    fn diversity_avoids_duplicates() {
        let cands: Vec<Cand> = (0..4)
            .map(|i| Cand {
                doc_idx: i,
                base: 1.0,
                text_len_chars: 40,
            })
            .collect();
        let sim = sim_map(&[(0, 1, 0.99), (0, 2, 0.98), (0, 3, 0.97)]);
        let pick = mmr_select(cands, 400, 3, 1.0, &sim);
        assert!(pick.contains(&0));
        assert_eq!(pick.len(), 3);
    }

    #[test]
    fn empty_candidates_ok() {
        let pick = mmr_select(vec![], 100, 10, 0.0, &|_, _| 0.0);
        assert!(pick.is_empty());
    }
}

