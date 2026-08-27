use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use crate::encoder::Encoder;
use crate::error::{Error, Result};
use crate::graph::{self, AssocGraph};
use crate::recall::{self, Cand, Profile, Weights};
use crate::salience::{self, recency, Tier};
use crate::sketch_hnsw::{SketchHnsw, GRAPH_THRESHOLD};
use crate::storage::wal::{WalOp, WalWriter};
use crate::storage::{
    self, new_segment_name, read_manifest, read_snapshot, write_manifest_atomic, write_snapshot,
    Manifest, SnapshotRow, WAL_FILE,
};
use crate::types::{estimate_tokens, Record, RememberOpts, StoredDoc};
use crate::unix_now;

const DEFAULT_HALF_LIFE: f64 = 30.0 * 86400.0;
const COLD_FALLBACK_FILL: usize = 16;
const MAX_SOURCES_PER_SUMMARY: usize = 5;

#[derive(Clone, Debug)]
pub struct Hit {
    pub id: u64,
    pub text: String,
    pub score: f32,
    pub semantic: f32,
    pub lexical: f32,
    pub recency: f32,
    pub importance: f32,
    pub estimated_tokens: usize,
    pub tier: Tier,
    pub sources: Vec<String>,
}

pub struct RecallOpts {
    pub query: String,
    pub budget_tokens: usize,
    pub k_max: usize,
    pub profile: Profile,
    pub weights: Option<Weights>,
    pub as_of: Option<i64>,
    pub half_life_secs: f64,
    pub include_cold: bool,
    pub expand_summaries: bool,
}

impl RecallOpts {
    pub fn new(query: impl Into<String>) -> Self {
        RecallOpts {
            query: query.into(),
            budget_tokens: 512,
            k_max: 64,
            profile: Profile::Chat,
            weights: None,
            as_of: None,
            half_life_secs: DEFAULT_HALF_LIFE,
            include_cold: false,
            expand_summaries: false,
        }
    }
    pub fn budget_tokens(mut self, t: usize) -> Self {
        self.budget_tokens = t.max(1);
        self
    }
    pub fn k_max(mut self, k: usize) -> Self {
        self.k_max = k.max(1);
        self
    }
    pub fn profile(mut self, p: Profile) -> Self {
        self.profile = p;
        self
    }
    pub fn weights(mut self, w: Weights) -> Self {
        self.weights = Some(w);
        self
    }
    pub fn as_of(mut self, t: i64) -> Self {
        self.as_of = Some(t);
        self
    }
    pub fn half_life_secs(mut self, s: f64) -> Self {
        self.half_life_secs = s.max(1.0);
        self
    }
    pub fn include_cold(mut self, v: bool) -> Self {
        self.include_cold = v;
        self
    }
    pub fn expand_summaries(mut self, v: bool) -> Self {
        self.expand_summaries = v;
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConsolidateCfg {
    pub cold_max_salience: f32,
    pub min_cluster: usize,
}

impl Default for ConsolidateCfg {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsolidateCfg {
    pub fn new() -> Self {
        ConsolidateCfg {
            cold_max_salience: 0.33,
            min_cluster: 3,
        }
    }
    pub fn min_cluster(mut self, n: usize) -> Self {
        self.min_cluster = n.max(2);
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConsolidateReport {
    pub clusters: usize,
    pub archived: usize,
    pub summaries_created: usize,
}

#[derive(Clone, Debug, Default)]
pub struct DetailedStats {
    pub live: usize,
    pub total_incl_dead: usize,
    pub archived: usize,
    pub summaries: usize,
    pub hot: usize,
    pub warm: usize,
    pub cold: usize,
}

struct Inner {
    docs: HashMap<u64, StoredDoc>,
    archive: HashMap<u64, StoredDoc>,
    subjects: HashMap<String, Vec<u64>>,
    sk_index: Option<SketchHnsw>,
    hot_tail: std::collections::VecDeque<u64>,
    df: HashMap<u64, u32>,
    next_id: u64,
    seg_seq: u64,
    wal: WalWriter,
    graph: AssocGraph,
}

pub struct Collection {
    pub name: String,
    dir: PathBuf,
    enc: Encoder,
    inner: Mutex<Inner>,
}

impl Collection {
    pub(crate) fn open(dir: PathBuf, name: String) -> Result<Collection> {
        fs::create_dir_all(&dir)?;
        let manifest = read_manifest(&dir)?;
        let mut docs: HashMap<u64, StoredDoc> = HashMap::new();
        let mut archive: HashMap<u64, StoredDoc> = HashMap::new();
        let mut subjects: HashMap<String, Vec<u64>> = HashMap::new();
        let mut pending_df: Vec<String> = Vec::new();
        let mut max_id = 0u64;

        for snap in &manifest.snapshots {
            for row in read_snapshot(&dir, snap)? {
                route_row(row, &mut docs, &mut archive, &mut subjects, &mut pending_df, &mut max_id);
            }
        }

        for arch in &manifest.wal_archives {
            let p = dir.join(arch);
            storage::wal::replay(&p, |op| match op {
                WalOp::Remember { row } => route_row(
                    row,
                    &mut docs,
                    &mut archive,
                    &mut subjects,
                    &mut pending_df,
                    &mut max_id,
                ),
                WalOp::Forget { id, at } => {
                    if let Some(d) = docs.get_mut(&id) {
                        d.record.valid_to = Some(at);
                    } else if let Some(d) = archive.get_mut(&id) {
                        d.record.valid_to = Some(at);
                    }
                }
                WalOp::Touch { id, at } => {
                    if let Some(d) = docs.get_mut(&id) {
                        d.record.hits += 1;
                        d.record.last_hit = Some(at);
                    } else if let Some(d) = archive.get_mut(&id) {
                        d.record.hits += 1;
                        d.record.last_hit = Some(at);
                    }
                }
                WalOp::HardDelete { id } => {
                    let removed = docs.remove(&id).or_else(|| archive.remove(&id));
                    if let Some(d) = removed {
                        if let Some(s) = &d.record.subject {
                            if let Some(v) = subjects.get_mut(s) {
                                v.retain(|&x| x != id);
                            }
                        }
                    }
                }
                WalOp::Archive { id } => {
                    if let Some(d) = docs.remove(&id) {
                        archive.insert(id, d);
                    }
                }
            })?;
        }

        storage::wal::replay(&dir.join(WAL_FILE), |op| match op {            WalOp::Remember { row } => {
                route_row(row, &mut docs, &mut archive, &mut subjects, &mut pending_df, &mut max_id)
            }
            WalOp::Forget { id, at } => {
                if let Some(d) = docs.get_mut(&id) {
                    d.record.valid_to = Some(at);
                } else if let Some(d) = archive.get_mut(&id) {
                    d.record.valid_to = Some(at);
                }
            }
            WalOp::Touch { id, at } => {
                if let Some(d) = docs.get_mut(&id) {
                    d.record.hits += 1;
                    d.record.last_hit = Some(at);
                } else if let Some(d) = archive.get_mut(&id) {
                    d.record.hits += 1;
                    d.record.last_hit = Some(at);
                }
            }
            WalOp::HardDelete { id } => {
                let removed = docs.remove(&id).or_else(|| archive.remove(&id));
                if let Some(d) = removed {
                    if let Some(s) = &d.record.subject {
                        if let Some(v) = subjects.get_mut(s) {
                            v.retain(|&x| x != id);
                        }
                    }
                }
            }
            WalOp::Archive { id } => {
                if let Some(d) = docs.remove(&id) {
                    archive.insert(id, d);
                }
            }
        })?;

        let next_id = manifest.next_id.max(max_id + 1);
        let seg_seq = manifest
            .snapshots
            .last()
            .and_then(|s| parse_seg_seq(s))
            .unwrap_or(0);

        let wal = WalWriter::open_append(&dir.join(WAL_FILE))?;
        let mut df = manifest.df;
        for text in &pending_df {
            Encoder::update_df(&mut df, text, 1);
        }

        let ndocs = docs.len() as u32;
        let mut graph = AssocGraph::default();
        for d in docs.values() {
            let kws = graph::graph_keywords_for(&d.record, &df, ndocs);
            graph.add_doc(d.record.id, &kws);
        }

        Ok(Collection {
            name,
            dir,
            enc: Encoder::new(),
            inner: Mutex::new(Inner {
                docs,
                archive,
                subjects,
                sk_index: None,
                hot_tail: std::collections::VecDeque::new(),
                df,
                next_id,
                seg_seq,
                wal,
                graph,
            }),
        })
    }

    pub fn remember(&self, opts: RememberOpts) -> Result<u64> {
        if opts.text.trim().is_empty() {
            return Err(Error::invalid("text must not be empty"));
        }
        let now = unix_now();
        let event_time = opts.event_time.unwrap_or(now);
        let mut inner = self.inner.lock().unwrap();
        ensure_graph(&mut inner);

        let encoded = self.enc.encode(&opts.text, &inner.df, inner.docs.len() as u32);
        let kws = graph::top_keywords(
            &opts.text,
            &inner.df,
            inner.docs.len() as u32,
            crate::graph::KEYWORDS_PER_DOC,
        );

        if let Some(subj) = &opts.subject {
            let live_ids: Vec<u64> = inner.subjects.get(subj).cloned().unwrap_or_default();
            for id in live_ids {
                let hit = if inner.docs.get(&id).map_or(false, |d| d.record.is_live()) {
                    inner.docs.get_mut(&id).map(|d| {
                        d.record.valid_to = Some(now);
                    })
                } else if inner.archive.get(&id).map_or(false, |d| d.record.is_live()) {
                    inner.archive.get_mut(&id).map(|d| {
                        d.record.valid_to = Some(now);
                    })
                } else {
                    None
                };
                if hit.is_some() {
                    inner.wal.append(&WalOp::Forget { id, at: now })?;
                }
            }
        }

        let id = inner.next_id;
        inner.next_id += 1;

        let record = Record {
            id,
            text: opts.text.clone(),
            subject: opts.subject.clone(),
            tags: opts.tags,
            event_time,
            ingest_time: now,
            valid_to: None,
            importance: opts.importance.clamp(0.0, 1.0),
            hits: 0,
            last_hit: None,
            source_ids: Vec::new(),
        };

        let row = SnapshotRow {
            record: record.clone(),
            lex: encoded.lex.clone(),
            sk: encoded.sketch.clone(),
            archived: false,
        };
        inner.wal.append(&WalOp::Remember { row })?;

        Encoder::update_df(&mut inner.df, &opts.text, 1);
        if let Some(s) = &record.subject {
            inner.subjects.entry(s.clone()).or_default().push(id);
        }
        inner.graph.add_doc(id, &kws);

        ensure_graph(&mut inner);
        if let Some(g) = &mut inner.sk_index {
            g.insert(id, &encoded.sketch);
        }
        const HOT_TAIL_CAP: usize = 512;
        if inner.hot_tail.len() >= HOT_TAIL_CAP {
            inner.hot_tail.pop_front();
        }
        inner.hot_tail.push_back(id);

        inner.docs.insert(
            id,
            StoredDoc {
                record,
                lex: encoded.lex,
                sk: encoded.sketch,
            },
        );
        Ok(id)
    }

    pub fn recall(&self, opts: RecallOpts) -> Result<Vec<Hit>> {
        if opts.query.trim().is_empty() {
            return Err(Error::invalid("query must not be empty"));
        }
        let now = unix_now();
        let as_of = opts.as_of.unwrap_or(now);
        let weights = opts.weights.unwrap_or_else(|| opts.profile.weights());
        let mut inner = self.inner.lock().unwrap();
        ensure_graph(&mut inner);

        let q = self.enc.encode(&opts.query, &inner.df, inner.docs.len() as u32);

        let mut view: Vec<&StoredDoc> = Vec::new();
        let mut cold_pool: Vec<&StoredDoc> = Vec::new();
        let mut excluded_cold: HashSet<u64> = HashSet::new();

        const GRAPH_POOL: usize = 192;
        let graph_ids: Vec<u64> = match &inner.sk_index {
            Some(g) => g
                .search(
                    &q.sketch,
                    GRAPH_POOL,
                    |gid| {
                        inner
                            .docs
                            .get(&gid)
                            .map_or(false, |d| d.record.live_at(as_of))
                    },
                )
                .into_iter()
                .map(|(id, _)| id)
                .collect(),
            None => Vec::new(),
        };
        let use_graph = !graph_ids.is_empty();
        let mut cand_ids: HashSet<u64> = graph_ids.iter().copied().collect();

        if use_graph {
            for &hid in inner.hot_tail.iter().rev().take(512) {
                cand_ids.insert(hid);
            }
            for gid in &cand_ids {
                if let Some(d) = inner.docs.get(gid) {
                    if d.record.live_at(as_of) {
                        classify_doc(
                            d,
                            &mut view,
                            &mut cold_pool,
                            &mut excluded_cold,
                            opts.profile,
                            opts.include_cold,
                            opts.half_life_secs,
                            as_of,
                        );
                    }
                }
            }
            if opts.profile == Profile::Overview && !view.iter().any(|d| d.record.is_summary()) {
                cand_ids.clear();
                view.clear();
                cold_pool.clear();
                excluded_cold.clear();
            }
        }

        if !use_graph || cand_ids.is_empty() {
            for d in inner.docs.values() {
                if !d.record.live_at(as_of) {
                    continue;
                }
                classify_doc(
                    d,
                    &mut view,
                    &mut cold_pool,
                    &mut excluded_cold,
                    opts.profile,
                    opts.include_cold,
                    opts.half_life_secs,
                    as_of,
                );
            }
        }
        if view.is_empty() && !cold_pool.is_empty() && !opts.include_cold {
            cold_pool.sort_by(|a, b| {
                b.record
                    .importance
                    .partial_cmp(&a.record.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let take = COLD_FALLBACK_FILL.min(cold_pool.len());
            view.extend(cold_pool.into_iter().take(take));
        }

        let q_kws = graph::top_keywords(
            &opts.query,
            &inner.df,
            inner.docs.len() as u32,
            crate::graph::KEYWORDS_PER_DOC,
        );
        let seen: HashSet<u64> = view.iter().map(|d| d.record.id).collect();
        let expanded = inner.graph.expand_candidates(&q_kws, |id| {
            !seen.contains(&id)
                && !excluded_cold.contains(&id)
                && inner
                    .docs
                    .get(&id)
                    .map_or(false, |d| d.record.live_at(as_of))
        });
        let mut bonus: HashMap<u64, f32> = HashMap::new();
        for (&doc_id, &aff) in expanded.iter() {
            if let Some(d) = inner.docs.get(&doc_id) {
                bonus.insert(doc_id, aff);
                view.push(d);
            }
        }

        let mut cands: Vec<Cand> = Vec::with_capacity(view.len());
        for (doc_idx, d) in view.iter().enumerate() {
            let semantic = d.sk.dot(&q.sketch);
            let lexical = d.lex.dot(&q.lex);
            let r = recency(as_of, d.record.event_time, opts.half_life_secs);
            let aff = bonus.get(&d.record.id).copied().unwrap_or(0.0);
            let base = recall::fuse(&weights, semantic, lexical, r, d.record.importance, aff);
            cands.push(Cand {
                doc_idx,
                base,
                text_len_chars: d.record.text.chars().count(),
            });
        }

        const CAND_POOL_CAP: usize = 256;
        if cands.len() > CAND_POOL_CAP {
            cands.select_nth_unstable_by(CAND_POOL_CAP - 1, |a, b| {
                b.base
                    .partial_cmp(&a.base)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            cands.truncate(CAND_POOL_CAP);
        }

        let picked = recall::mmr_select(
            cands,
            opts.budget_tokens,
            opts.k_max,
            weights.diversity,
            &|a, b| view[a].sk.dot(&view[b].sk),
        );

        let picked_meta: Vec<Hit> = picked
            .iter()
            .map(|&idx| {
                let d = view[idx];
                let semantic = d.sk.dot(&q.sketch);
                let lexical = d.lex.dot(&q.lex);
                let r = recency(as_of, d.record.event_time, opts.half_life_secs);
                let aff = bonus.get(&d.record.id).copied().unwrap_or(0.0);
                let score = recall::fuse(&weights, semantic, lexical, r, d.record.importance, aff);
                let s = salience::salience(&d.record, as_of, opts.half_life_secs);
                Hit {
                    id: d.record.id,
                    text: d.record.text.clone(),
                    score,
                    semantic,
                    lexical,
                    recency: r,
                    importance: d.record.importance,
                    estimated_tokens: estimate_tokens(&d.record.text),
                    tier: salience::tier(s),
                    sources: Vec::new(),
                }
            })
            .collect();

        let mut hits = Vec::with_capacity(picked_meta.len());
        for mut h in picked_meta {
            if let Some(live) = inner.docs.get_mut(&h.id) {
                live.record.hits += 1;
                live.record.last_hit = Some(now);
            }
            inner.wal.append(&WalOp::Touch { id: h.id, at: now })?;

            if opts.expand_summaries {
                let src_ids = inner
                    .docs
                    .get(&h.id)
                    .map(|d| d.record.source_ids.clone())
                    .unwrap_or_default();
                for sid in src_ids.iter().take(MAX_SOURCES_PER_SUMMARY) {
                    for m in [&inner.docs, &inner.archive] {
                        if let Some(sd) = m.get(sid) {
                            h.sources.push(sd.record.text.clone());
                            break;
                        }
                    }
                }
            }
            hits.push(h);
        }
        Ok(hits)
    }

    pub fn forget(&self, id: u64) -> Result<()> {
        let now = unix_now();
        let mut inner = self.inner.lock().unwrap();
        if let Some(d) = inner.docs.get_mut(&id) {
            d.record.valid_to = Some(now);
            return inner.wal.append(&WalOp::Forget { id, at: now });
        }
        if let Some(d) = inner.archive.get_mut(&id) {
            d.record.valid_to = Some(now);
            return inner.wal.append(&WalOp::Forget { id, at: now });
        }
        Err(Error::NotFound(id))
    }

    pub fn forget_subject(&self, subject: &str) -> Result<usize> {
        let now = unix_now();
        let mut inner = self.inner.lock().unwrap();
        let ids: Vec<u64> = inner.subjects.get(subject).cloned().unwrap_or_default();
        let mut n = 0;
        for id in ids {
            let hit_live = if inner.docs.get(&id).map_or(false, |d| d.record.is_live()) {
                if let Some(d) = inner.docs.get_mut(&id) {
                    d.record.valid_to = Some(now);
                }
                true
            } else if inner.archive.get(&id).map_or(false, |d| d.record.is_live()) {
                if let Some(d) = inner.archive.get_mut(&id) {
                    d.record.valid_to = Some(now);
                }
                true
            } else {
                false
            };
            if hit_live {
                inner.wal.append(&WalOp::Forget { id, at: now })?;
                n += 1;
            }
        }
        Ok(n)
    }

    pub fn hard_delete(&self, id: u64) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let removed = if inner.docs.contains_key(&id) {
            inner.docs.remove(&id)
        } else {
            inner.archive.remove(&id)
        };
        match removed {
            Some(d) => {
                if let Some(s) = &d.record.subject {
                    if let Some(v) = inner.subjects.get_mut(s) {
                        v.retain(|&x| x != id);
                    }
                }
                let kws = graph::graph_keywords_for(&d.record, &inner.df, inner.docs.len() as u32);
                inner.graph.remove_doc(id, &kws);
                Encoder::update_df(&mut inner.df, &d.record.text, -1);
                inner.wal.append(&WalOp::HardDelete { id })
            }
            None => Err(Error::NotFound(id)),
        }
    }

    pub fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock().unwrap();
        let live = inner.docs.values().filter(|d| d.record.is_live()).count();
        (live, inner.docs.len())
    }

    pub fn detailed_stats(&self) -> DetailedStats {
        let inner = self.inner.lock().unwrap();
        let now = unix_now();
        let mut st = DetailedStats {
            total_incl_dead: inner.docs.len(),
            archived: inner.archive.len(),
            ..Default::default()
        };
        for d in inner.docs.values() {
            if !d.record.is_live() {
                continue;
            }
            if d.record.is_summary() {
                st.summaries += 1;
            }
            match salience::tier(salience::salience(&d.record, now, DEFAULT_HALF_LIFE)) {
                Tier::Hot => st.hot += 1,
                Tier::Warm => st.warm += 1,
                Tier::Cold => st.cold += 1,
            }
        }
        st.live = st.hot + st.warm + st.cold;
        st
    }

    pub fn consolidate(&self, cfg: ConsolidateCfg) -> Result<ConsolidateReport> {
        let now = unix_now();
        let mut report = ConsolidateReport {
            clusters: 0,
            archived: 0,
            summaries_created: 0,
        };
        {
            let mut inner = self.inner.lock().unwrap();
            let mut groups: HashMap<String, Vec<u64>> = HashMap::new();
            for d in inner.docs.values() {
                if !d.record.is_live()
                    || d.record.is_summary()
                    || d.record.hits > 0
                    || salience::salience(&d.record, now, DEFAULT_HALF_LIFE) > cfg.cold_max_salience
                {
                    continue;
                }
                let key = graph::top_keywords(&d.record.text, &inner.df, inner.docs.len() as u32, 1)
                    .into_iter()
                    .next()
                    .or_else(|| d.record.subject.clone());
                if let Some(kw) = key {
                    groups.entry(kw).or_default().push(d.record.id);
                }
            }

            let ndocs = inner.docs.len() as u32;
            #[cfg(test)]
            for (kw, ms) in &groups {
                eprintln!("[consolidate-probe] kw={kw:?} members={}", ms.len());
            }
            for (kw, mut members) in groups {
                if members.len() < cfg.min_cluster {
                    continue;
                }
                members.sort_by(|&a, &b| {
                    let sa = inner
                        .docs
                        .get(&a)
                        .map(|d| salience::salience(&d.record, now, DEFAULT_HALF_LIFE))
                        .unwrap_or(0.0);
                    let sb = inner
                        .docs
                        .get(&b)
                        .map(|d| salience::salience(&d.record, now, DEFAULT_HALF_LIFE))
                        .unwrap_or(0.0);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });

                let texts: Vec<String> = members
                    .iter()
                    .filter_map(|id| inner.docs.get(id))
                    .map(|d| clip(&d.record.text, 80))
                    .collect();
                let summary_text = format!("【摘要】关于“{}”的 {} 条记忆：{}", kw, members.len(), texts.join(" / "));
                let importance = members
                    .iter()
                    .filter_map(|id| inner.docs.get(id))
                    .map(|d| d.record.importance)
                    .fold(0.0f32, |a, v| a.max(v));
                let event_time = members
                    .iter()
                    .filter_map(|id| inner.docs.get(id))
                    .map(|d| d.record.event_time)
                    .max()
                    .unwrap_or(now);

                let encoded = self.enc.encode(&summary_text, &inner.df, ndocs);
                let sid = inner.next_id;
                inner.next_id += 1;
                let record = Record {
                    id: sid,
                    text: summary_text.clone(),
                    subject: None,
                    tags: vec!["summary".to_string(), kw],
                    event_time,
                    ingest_time: now,
                    valid_to: None,
                    importance,
                    hits: 0,
                    last_hit: None,
                    source_ids: members.clone(),
                };
                let kws = graph::top_keywords(&summary_text, &inner.df, ndocs, crate::graph::KEYWORDS_PER_DOC);
                inner.graph.add_doc(sid, &kws);

                for &cid in &members {
                    if let Some(child) = inner.docs.remove(&cid) {
                        let ckws =
                            graph::graph_keywords_for(&child.record, &inner.df, ndocs);
                        inner.graph.remove_doc(cid, &ckws);
                        inner.archive.insert(cid, child);
                    }
                    inner.wal.append(&WalOp::Archive { id: cid })?;
                }
                report.archived += members.len();

                let row = SnapshotRow {
                    record: record.clone(),
                    lex: encoded.lex.clone(),
                    sk: encoded.sketch.clone(),
                    archived: false,
                };
                inner.wal.append(&WalOp::Remember { row })?;
                inner.docs.insert(
                    sid,
                    StoredDoc {
                        record,
                        lex: encoded.lex,
                        sk: encoded.sketch,
                    },
                );
                report.clusters += 1;
                report.summaries_created += 1;
            }
        }
        if report.clusters > 0 {
            self.checkpoint()?;
        }
        Ok(report)
    }

    pub fn checkpoint(&self) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.seg_seq += 1;
        let name = new_segment_name(inner.seg_seq);
        let live_rows = inner.docs.values().map(|d| SnapshotRow {
            record: d.record.clone(),
            lex: d.lex.clone(),
            sk: d.sk.clone(),
            archived: false,
        });
        let arch_rows = inner.archive.values().map(|d| SnapshotRow {
            record: d.record.clone(),
            lex: d.lex.clone(),
            sk: d.sk.clone(),
            archived: true,
        });
        write_snapshot(&self.dir, &name, live_rows.chain(arch_rows))?;
        let m = Manifest {
            version: 1,
            next_id: inner.next_id,
            snapshots: vec![name],
            df: inner.df.clone(),
            wal_archives: Vec::new(),
        };
        write_manifest_atomic(&self.dir, &m)?;
        storage::cleanup_unlisted_segments(&self.dir, &m.snapshots)?;
        inner.wal = WalWriter::create(&self.dir.join(WAL_FILE))?;
        Ok(())
    }

    /// PITR-friendly checkpoint: archive the current WAL instead of
    /// truncating it. Recovery replays archived segments after the snapshot.
    pub fn checkpoint_keep_wal(&self) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.seg_seq += 1;
        let name = new_segment_name(inner.seg_seq);
        let live_rows = inner.docs.values().map(|d| SnapshotRow {
            record: d.record.clone(),
            lex: d.lex.clone(),
            sk: d.sk.clone(),
            archived: false,
        });
        let arch_rows = inner.archive.values().map(|d| SnapshotRow {
            record: d.record.clone(),
            lex: d.lex.clone(),
            sk: d.sk.clone(),
            archived: true,
        });
        write_snapshot(&self.dir, &name, live_rows.chain(arch_rows))?;

        let mut m = Manifest {
            version: 1,
            next_id: inner.next_id,
            snapshots: vec![name],
            df: inner.df.clone(),
            wal_archives: Vec::new(),
        };
        {
            let old = read_manifest(&self.dir)?;
            m.wal_archives = old.wal_archives;
        }
        let wal_path = self.dir.join(WAL_FILE);
        if wal_path.exists() {
            let arch_name = format!("wal-a{:06}-{}.archived", inner.seg_seq, unix_now());
            fs::rename(&wal_path, self.dir.join(&arch_name))?;
            m.wal_archives.push(arch_name);
        }
        write_manifest_atomic(&self.dir, &m)?;
        storage::cleanup_unlisted_segments(&self.dir, &m.snapshots)?;
        inner.wal = WalWriter::create(&wal_path)?;
        Ok(())
    }

    pub(crate) fn dir_path(&self) -> &PathBuf {
        &self.dir
    }

    /// Ids of live summary nodes (introspection for maintenance tooling).
    pub fn summary_ids(&self) -> Vec<u64> {
        let inner = self.inner.lock().unwrap();
        inner
            .docs
            .values()
            .filter(|d| d.record.is_summary() && d.record.is_live())
            .map(|d| d.record.id)
            .collect()
    }

    /// Insert a prebuilt snapshot row (export/import path). Keeps the original
    /// id; returns false when the id already exists (idempotent import).
    pub fn import_row(&self, row: SnapshotRow) -> Result<bool> {
        let mut inner = self.inner.lock().unwrap();
        let id = row.record.id;
        if inner.docs.contains_key(&id) || inner.archive.contains_key(&id) {
            return Ok(false);
        }
        if id >= inner.next_id {
            inner.next_id = id + 1;
        }
        let encoded_lex = row.lex;
        let encoded_sk = row.sk;
        let text = row.record.text.clone();

        inner.wal.append(&WalOp::Remember {
            row: SnapshotRow {
                record: row.record.clone(),
                lex: encoded_lex.clone(),
                sk: encoded_sk.clone(),
                archived: row.archived,
            },
        })?;
        Encoder::update_df(&mut inner.df, &text, 1);
        if let Some(s) = &row.record.subject {
            inner.subjects.entry(s.clone()).or_default().push(id);
        }
        let kws = graph::top_keywords(
            &text,
            &inner.df,
            inner.docs.len() as u32,
            crate::graph::KEYWORDS_PER_DOC,
        );
        inner.graph.add_doc(id, &kws);

        let target = if row.archived {
            &mut inner.archive
        } else {
            &mut inner.docs
        };
        target.insert(
            id,
            StoredDoc {
                record: row.record,
                lex: encoded_lex,
                sk: encoded_sk,
            },
        );
        Ok(true)
    }

    /// Drop dead records older than `retention_secs`, never touching anything
    /// referenced by a summary's source_ids. Ends with a checkpoint.
    pub fn compact(&self, retention_secs: u64) -> Result<(usize, usize)> {
        let now = unix_now();
        let mut removed_live_dead = 0usize;
        let mut removed_archived = 0usize;
        {
            let mut inner = self.inner.lock().unwrap();

            let expired_valid_to = |valid_to: Option<i64>| -> bool {
                match valid_to {
                    Some(v) => now.saturating_sub(v) > retention_secs as i64,
                    None => false,
                }
            };

            let mut referenced: HashSet<u64> = HashSet::new();
            // A LIVE (not yet forgotten) summary grants protection to its
            // source history; once the summary itself is forgotten/compacted
            // away, its sources age out independently.
            for d in inner.docs.values() {
                if d.record.is_summary()
                    && d.record.is_live()
                    && !expired_valid_to(d.record.valid_to)
                {
                    referenced.extend(d.record.source_ids.iter().copied());
                }
            }
            for d in inner.archive.values() {
                if d.record.is_summary()
                    && d.record.is_live()
                    && !expired_valid_to(d.record.valid_to)
                {
                    referenced.extend(d.record.source_ids.iter().copied());
                }
            }

            let expired_valid_to = |valid_to: Option<i64>| -> bool {
                match valid_to {
                    Some(v) => now.saturating_sub(v) > retention_secs as i64,
                    None => false,
                }
            };
            // Archived payloads carry no valid_to; their age is judged by
            // event_time so retention bounds total history depth.
            let expired_history = |r: &crate::types::Record| -> bool {
                now.saturating_sub(r.event_time.max(0)) > retention_secs as i64
            };

            let dead_ids: Vec<u64> = inner
                .docs
                .iter()
                .filter(|(_, d)| {
                    !d.record.is_summary()
                        && !referenced.contains(&d.record.id)
                        && expired_valid_to(d.record.valid_to)
                })
                .map(|(id, _)| *id)
                .collect();
            for id in dead_ids {
                if let Some(d) = inner.docs.remove(&id) {
                    let kws =
                        graph::graph_keywords_for(&d.record, &inner.df, inner.docs.len() as u32);
                    inner.graph.remove_doc(id, &kws);
                    removed_live_dead += 1;
                }
            }

            let dead_arch: Vec<u64> = inner
                .archive
                .iter()
                .filter(|(_, d)| {
                    !d.record.is_summary()
                        && !referenced.contains(&d.record.id)
                        && (expired_valid_to(d.record.valid_to)
                            || expired_history(&d.record))
                })
                .map(|(id, _)| *id)
                .collect();
            for id in dead_arch {
                if inner.archive.remove(&id).is_some() {
                    removed_archived += 1;
                }
            }
        }
        self.checkpoint()?;
        Ok((removed_live_dead, removed_archived))
    }
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "…"
    }
}

#[allow(clippy::too_many_arguments)]
fn classify_doc<'a>(
    d: &'a StoredDoc,
    view: &mut Vec<&'a StoredDoc>,
    cold_pool: &mut Vec<&'a StoredDoc>,
    excluded_cold: &mut HashSet<u64>,
    profile: Profile,
    include_cold: bool,
    half_life: f64,
    as_of: i64,
) {
    let s = salience::salience(&d.record, as_of, half_life);
    if profile == Profile::Overview && d.record.is_summary() {
        view.push(d);
        return;
    }
    match salience::tier(s) {
        Tier::Cold if !include_cold => {
            excluded_cold.insert(d.record.id);
            cold_pool.push(d);
        }
        _ => view.push(d),
    }
}

fn ensure_graph(inner: &mut Inner) {
    if inner.sk_index.is_some() {
        return;
    }
    let live = inner.docs.values().filter(|d| d.record.is_live()).count();
    if live < GRAPH_THRESHOLD.load(Ordering::Relaxed) {
        return;
    }
    let mut g = SketchHnsw::new();
    for d in inner.docs.values() {
        if d.record.is_live() && !d.sk.is_empty() {
            g.insert(d.record.id, &d.sk);
        }
    }
    let mut newest: Vec<(i64, u64)> = inner
        .docs
        .values()
        .filter(|d| d.record.is_live())
        .map(|d| (d.record.ingest_time, d.record.id))
        .collect();
    newest.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    for (_, id) in newest.into_iter().take(512) {
        inner.hot_tail.push_back(id);
    }
    inner.sk_index = Some(g);
}

fn route_row(
    row: SnapshotRow,
    docs: &mut HashMap<u64, StoredDoc>,
    archive: &mut HashMap<u64, StoredDoc>,
    subjects: &mut HashMap<String, Vec<u64>>,
    pending_df: &mut Vec<String>,
    max_id: &mut u64,
) {
    *max_id = (*max_id).max(row.record.id);
    if let Some(s) = &row.record.subject {
        subjects.entry(s.clone()).or_default().push(row.record.id);
    }
    pending_df.push(row.record.text.clone());
    let target = if row.archived { archive } else { docs };
    target.insert(
        row.record.id,
        StoredDoc {
            record: row.record,
            lex: row.lex,
            sk: row.sk,
        },
    );
}

fn parse_seg_seq(name: &str) -> Option<u64> {
    let stem = name.strip_prefix("seg-")?.strip_suffix(".jsonl")?;
    stem.parse().ok()
}

