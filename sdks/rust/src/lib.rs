//! EngramDB Rust SDK.
//!
//! Facade over the `engram-db` engine: a stable, curated surface plus
//! ergonomic extension traits so agent code can create databases and use
//! memory without touching builder structs.
//!
//! ```no_run
//! use engram::prelude::*;
//!
//! let db = engram::open("./memory")?;
//! let col = db.create_collection("assistant")?;
//! col.remember_text("user prefers dark theme")?;
//! let block = col.recall_for_prompt("theme preference", 300)?;
//! println!("{block}");
//! # Ok::<(), engram::Error>(())
//! ```

pub use engram_db::collection::{
    ConsolidateCfg, ConsolidateReport, DetailedStats, Hit, RecallOpts,
};
pub use engram_db::db::Db;
pub use engram_db::index;
pub use engram_db::recall::Profile;
pub use engram_db::salience::Tier;
pub use engram_db::{Error, Record, RememberOpts, Result};

pub mod prelude {
    pub use crate::{
        DatabaseMaintenanceExt, Db, Hit, HitFmt, Profile, RecallExt, RecallOpts,
        RememberExt, RememberOpts, Tier,
    };
    pub use crate::{open, Record};
}

/// Open (or create) a memory database rooted at `path`.
///
/// Creation is implicit: directories are made on demand and an exclusive
/// cross-process lock is taken for the lifetime of the handle.
pub fn open(path: impl AsRef<std::path::Path>) -> Result<Db> {
    Db::open(path)
}

/// One-call write conveniences on top of [`RememberOpts`].
pub trait RememberExt {
    /// Minimal write: text only, defaults for everything else.
    fn remember_text(&self, text: impl Into<String>) -> Result<u64>;

    /// Write bound to a subject — later writes on the same subject
    /// automatically supersede this record (bitemporal version chain).
    fn remember_about(
        &self,
        text: impl Into<String>,
        subject: impl Into<String>,
        importance: f32,
    ) -> Result<u64>;
}

impl RememberExt for engram_db::collection::Collection {
    fn remember_text(&self, text: impl Into<String>) -> Result<u64> {
        self.remember(RememberOpts::new(text))
    }

    fn remember_about(
        &self,
        text: impl Into<String>,
        subject: impl Into<String>,
        importance: f32,
    ) -> Result<u64> {
        self.remember(
            RememberOpts::new(text)
                .subject(subject)
                .importance(importance),
        )
    }
}

/// Recall conveniences tuned for LLM context assembly.
pub trait RecallExt {
    /// Sensible defaults: chat profile, 512-token budget.
    fn recall_simple(&self, query: &str) -> Result<Vec<Hit>>;

    /// Render memories as a prompt-ready markdown block sized to
    /// `budget_tokens`, with per-item source annotations when available.
    fn recall_for_prompt(&self, query: &str, budget_tokens: usize) -> Result<String>;
}

impl RecallExt for engram_db::collection::Collection {
    fn recall_simple(&self, query: &str) -> Result<Vec<Hit>> {
        self.recall(RecallOpts::new(query))
    }

    fn recall_for_prompt(&self, query: &str, budget_tokens: usize) -> Result<String> {
        let hits = self.recall(RecallOpts::new(query).budget_tokens(budget_tokens))?;
        if hits.is_empty() {
            return Ok(String::new());
        }
        let mut out = format!(
            "<memory budget={}B items={}\n",
            budget_tokens,
            hits.len()
        );
        for h in &hits {
            out.push_str(&format!(
                "- [{}] {}\n",
                h.tier_short(),
                h.text.replace('\n', " ")
            ));
        }
        out.push_str("</memory>");
        Ok(out)
    }
}

/// Short tier tag for prompt rendering ("H"/"W"/"C").
pub trait HitFmt {
    fn tier_short(&self) -> &'static str;
}

impl HitFmt for Hit {
    fn tier_short(&self) -> &'static str {
        match self.tier {
            Tier::Hot => "H",
            Tier::Warm => "W",
            Tier::Cold => "C",
        }
    }
}

/// Maintenance operations mirroring the engine's data-safety suite.
pub trait DatabaseMaintenanceExt {
    /// Consistent online backup; returns the destination path.
    fn backup(&self, dest: impl AsRef<std::path::Path>) -> Result<std::path::PathBuf>;

    /// Restore a backup directory into a NEW database location.
    fn restore(
        src: impl AsRef<std::path::Path>,
        dest: impl AsRef<std::path::Path>,
    ) -> Result<Db>;

    /// Export every collection to NDJSON; returns row count.
    fn export_jsonl_to(&self, path: impl AsRef<std::path::Path>) -> Result<u64>;

    /// Import NDJSON rows; returns (imported, skipped).
    fn import_jsonl_from(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(u64, u64)>;

    /// Structural integrity check over all persisted rows.
    fn verify_report(&self) -> engram_db::db::VerifyReport;

    /// Drop dead records older than `retention_secs`; returns removal counts.
    fn compact(&self, retention_secs: u64) -> Result<(usize, usize)>;

    /// PITR-friendly checkpoint (archives WAL instead of truncating).
    fn checkpoint_keep_wal(&self) -> Result<()>;
}

impl DatabaseMaintenanceExt for Db {
    fn backup(&self, dest: impl AsRef<std::path::Path>) -> Result<std::path::PathBuf> {
        self.backup_to(dest)
    }

    fn restore(
        src: impl AsRef<std::path::Path>,
        dest: impl AsRef<std::path::Path>,
    ) -> Result<Db> {
        Db::restore_from(src, dest)
    }

    fn export_jsonl_to(&self, path: impl AsRef<std::path::Path>) -> Result<u64> {
        let f = std::fs::File::create(path)?;
        self.export_jsonl(f)
    }

    fn import_jsonl_from(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(u64, u64)> {
        let f = std::fs::File::open(path)?;
        self.import_jsonl(f)
    }

    fn verify_report(&self) -> engram_db::db::VerifyReport {
        self.verify()
    }

    fn compact(&self, retention_secs: u64) -> Result<(usize, usize)> {
        Db::compact(self, retention_secs)
    }

    fn checkpoint_keep_wal(&self) -> Result<()> {
        Db::checkpoint_all_keep_wal(self)
    }
}
