use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use engram_db::collection::{Collection as CoreCollection, Hit, RecallOpts};
use engram_db::db::Db as CoreDb;
use engram_db::{Profile, RememberOpts};

fn to_pyerr(e: engram_db::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pyclass]
struct Db {
    inner: CoreDb,
}

#[pymethods]
impl Db {
    #[new]
    #[pyo3(signature = (path))]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Db {
            inner: CoreDb::open(path).map_err(to_pyerr)?,
        })
    }

    fn create_collection(&self, name: &str) -> PyResult<Collection> {
        Ok(Collection {
            inner: self.inner.create_collection(name).map_err(to_pyerr)?,
        })
    }

    fn collection(&self, name: &str) -> PyResult<Collection> {
        Ok(Collection {
            inner: self.inner.collection(name).map_err(to_pyerr)?,
        })
    }

    fn collection_names(&self) -> Vec<String> {
        self.inner.collection_names()
    }

    fn checkpoint_all(&self) -> PyResult<()> {
        self.inner.checkpoint_all().map_err(to_pyerr)
    }

    /// PITR-friendly checkpoint: archives WAL instead of truncating.
    fn checkpoint_keep_wal(&self) -> PyResult<()> {
        self.inner.checkpoint_all_keep_wal().map_err(to_pyerr)
    }

    /// Consistent online backup into `dest` (creates it).
    fn backup_to(&self, dest: &str) -> PyResult<String> {
        let p = self.inner.backup_to(dest).map_err(to_pyerr)?;
        Ok(p.to_string_lossy().to_string())
    }

    /// Restore a backup directory into a NEW database at `dest`.
    #[staticmethod]
    fn restore_from(src: &str, dest: &str) -> PyResult<Db> {
        Ok(Db {
            inner: CoreDb::restore_from(src, dest).map_err(to_pyerr)?,
        })
    }

    /// Export all collections as NDJSON (ENGR-1 rows) to `path`.
    /// Returns the number of rows written.
    fn export_jsonl(&self, path: &str) -> PyResult<u64> {
        let f = std::fs::File::create(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.export_jsonl(f).map_err(to_pyerr)
    }

    /// Import rows previously written by export_jsonl. Idempotent on ids.
    /// Returns (imported, skipped).
    fn import_jsonl(&self, path: &str) -> PyResult<(u64, u64)> {
        let f = std::fs::File::open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.import_jsonl(f).map_err(to_pyerr)
    }

    /// Structural integrity check. Returns {"collections","rows","ok","errors"}.
    fn verify(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let r = self.inner.verify();
        let d = PyDict::new_bound(py);
        d.set_item("collections", r.collections)?;
        d.set_item("rows", r.rows)?;
        d.set_item("ok", r.ok())?;
        d.set_item("errors", r.errors)?;
        Ok(d.into())
    }

    /// Drop dead records older than `retention_secs`.
    /// Returns (live_dead_removed, archived_removed).
    fn compact(&self, retention_secs: u64) -> PyResult<(usize, usize)> {
        self.inner.compact(retention_secs).map_err(to_pyerr)
    }
}

/// Open (or create) a memory database rooted at `path`.
#[pyfunction]
fn open(path: &str) -> PyResult<Db> {
    Db::new(path)
}

#[pyclass(get_all)]
struct PyHit {
    id: u64,
    text: String,
    score: f32,
    semantic: f32,
    lexical: f32,
    recency: f32,
    importance: f32,
    estimated_tokens: usize,
    tier: String,
    sources: Vec<String>,
}

#[pymethods]
impl PyHit {
    fn __repr__(&self) -> String {
        format!(
            "Hit(id={}, score={:.3}, tier='{}', tokens=~{}, text={:?})",
            self.id, self.score, self.tier, self.estimated_tokens, self.text
        )
    }
}

fn hit_to_py(h: Hit) -> PyHit {
    PyHit {
        id: h.id,
        text: h.text,
        score: h.score,
        semantic: h.semantic,
        lexical: h.lexical,
        recency: h.recency,
        importance: h.importance,
        estimated_tokens: h.estimated_tokens,
        tier: format!("{:?}", h.tier),
        sources: h.sources,
    }
}

#[pyclass]
struct Collection {
    inner: Arc<CoreCollection>,
}

#[pymethods]
impl Collection {
    #[pyo3(signature = (text, subject=None, tags=Vec::new(), importance=0.5, event_time=None))]
    fn remember(
        &self,
        text: String,
        subject: Option<String>,
        tags: Vec<String>,
        importance: f32,
        event_time: Option<i64>,
    ) -> PyResult<u64> {
        let opts = RememberOpts::new(text)
            .importance(importance)
            .tags(tags)
            .event_time_opt(event_time);
        let opts = match subject {
            Some(s) => opts.subject(s),
            None => opts,
        };
        self.inner.remember(opts).map_err(to_pyerr)
    }

    #[pyo3(signature = (query, budget_tokens=512, k_max=64, profile="chat", half_life_secs=None, include_cold=false, expand_summaries=false))]
    fn recall(
        &self,
        query: &str,
        budget_tokens: usize,
        k_max: usize,
        profile: &str,
        half_life_secs: Option<f64>,
        include_cold: bool,
        expand_summaries: bool,
    ) -> PyResult<Vec<PyHit>> {
        let p = match profile {
            "agent" | "agent-task" => Profile::AgentTask,
            "overview" => Profile::Overview,
            _ => Profile::Chat,
        };
        let mut o = RecallOpts::new(query)
            .budget_tokens(budget_tokens)
            .k_max(k_max)
            .profile(p)
            .include_cold(include_cold)
            .expand_summaries(expand_summaries);
        if let Some(hl) = half_life_secs {
            o = o.half_life_secs(hl);
        }
        let hits = self.inner.recall(o).map_err(to_pyerr)?;
        Ok(hits.into_iter().map(hit_to_py).collect())
    }

    fn forget(&self, id: u64) -> PyResult<()> {
        self.inner.forget(id).map_err(to_pyerr)
    }

    fn forget_subject(&self, subject: &str) -> PyResult<usize> {
        self.inner.forget_subject(subject).map_err(to_pyerr)
    }

    fn hard_delete(&self, id: u64) -> PyResult<()> {
        self.inner.hard_delete(id).map_err(to_pyerr)
    }

    fn stats(&self) -> (usize, usize) {
        self.inner.stats()
    }

    fn detailed_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let st = self.inner.detailed_stats();
        let d = PyDict::new_bound(py);
        d.set_item("live", st.live)?;
        d.set_item("total_incl_dead", st.total_incl_dead)?;
        d.set_item("archived", st.archived)?;
        d.set_item("summaries", st.summaries)?;
        d.set_item("hot", st.hot)?;
        d.set_item("warm", st.warm)?;
        d.set_item("cold", st.cold)?;
        Ok(d.into())
    }

    #[pyo3(signature = (min_cluster=3))]
    fn consolidate(&self, min_cluster: usize) -> PyResult<(usize, usize, usize)> {
        let cfg = engram_db::collection::ConsolidateCfg::new().min_cluster(min_cluster);
        let r = self.inner.consolidate(cfg).map_err(to_pyerr)?;
        Ok((r.clusters, r.archived, r.summaries_created))
    }

    fn checkpoint(&self) -> PyResult<()> {
        self.inner.checkpoint().map_err(to_pyerr)
    }
}

#[pymodule]
fn engram(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_class::<Db>()?;
    m.add_class::<Collection>()?;
    m.add_class::<PyHit>()?;
    Ok(())
}

