use serde::{Deserialize, Serialize};

use crate::sketch::Sketch;
use crate::sparse::SparseVec;

pub const DIM: usize = 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub id: u64,
    pub text: String,
    pub subject: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub event_time: i64,
    pub ingest_time: i64,
    pub valid_to: Option<i64>,
    pub importance: f32,
    pub hits: u32,
    pub last_hit: Option<i64>,
    #[serde(default)]
    pub source_ids: Vec<u64>,
}

impl Record {
    pub fn live_at(&self, as_of: i64) -> bool {
        self.valid_to.map_or(true, |v| v > as_of)
    }
    pub fn is_live(&self) -> bool {
        self.valid_to.is_none()
    }
    pub fn is_summary(&self) -> bool {
        self.tags.iter().any(|t| t == "summary")
    }
}

#[derive(Clone, Debug)]
pub struct StoredDoc {
    pub record: Record,
    pub lex: SparseVec,
    pub sk: Sketch,
}

pub struct RememberOpts {
    pub text: String,
    pub subject: Option<String>,
    pub tags: Vec<String>,
    pub importance: f32,
    pub event_time: Option<i64>,
}

impl RememberOpts {
    pub fn new(text: impl Into<String>) -> Self {
        RememberOpts {
            text: text.into(),
            subject: None,
            tags: Vec::new(),
            importance: 0.5,
            event_time: None,
        }
    }
    pub fn subject(mut self, s: impl Into<String>) -> Self {
        self.subject = Some(s.into());
        self
    }
    pub fn tags(mut self, t: Vec<String>) -> Self {
        self.tags = t;
        self
    }
    pub fn importance(mut self, v: f32) -> Self {
        self.importance = v.clamp(0.0, 1.0);
        self
    }
    pub fn event_time(mut self, t: i64) -> Self {
        self.event_time = Some(t);
        self
    }
    pub fn event_time_opt(mut self, t: Option<i64>) -> Self {
        self.event_time = t;
        self
    }
}

pub const TOKEN_CHARS_PER_TOKEN: usize = 4;

pub fn estimate_tokens(s: &str) -> usize {
    (s.chars().count() + TOKEN_CHARS_PER_TOKEN - 1) / TOKEN_CHARS_PER_TOKEN
}

