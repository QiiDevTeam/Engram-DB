use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::collection::Collection;
use crate::error::{Error, Result};
use crate::storage::lock::FileLock;
use crate::storage::{SnapshotRow, MANIFEST_FILE};

const COLLECTIONS_DIR: &str = "collections";
const LOCK_FILE: &str = "engram.lock";

pub struct Db {
    root: PathBuf,
    cols: Mutex<HashMap<String, Arc<Collection>>>,
    _lock: FileLock,
}

impl Db {
    pub fn open(root: impl AsRef<Path>) -> Result<Db> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join(COLLECTIONS_DIR))?;
        let lock = FileLock::acquire(&root.join(LOCK_FILE))?;
        let mut cols = HashMap::new();
        let base = root.join(COLLECTIONS_DIR);
        for entry in fs::read_dir(&base)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if !valid_name(&name) {
                continue;
            }
            let col = Collection::open(entry.path(), name.clone())?;
            cols.insert(name, Arc::new(col));
        }
        Ok(Db {
            root,
            cols: Mutex::new(cols),
            _lock: lock,
        })
    }

    pub fn create_collection(&self, name: &str) -> Result<Arc<Collection>> {
        if !valid_name(name) {
            return Err(Error::invalid(
                "collection name must match [A-Za-z0-9_-]{1,64}",
            ));
        }
        let mut cols = self.cols.lock().unwrap();
        if let Some(c) = cols.get(name) {
            return Ok(c.clone());
        }
        let dir = self.root.join(COLLECTIONS_DIR).join(name);
        let col = Arc::new(Collection::open(dir, name.to_string())?);
        cols.insert(name.to_string(), col.clone());
        Ok(col)
    }

    pub fn collection(&self, name: &str) -> Result<Arc<Collection>> {
        self.cols
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::CollectionNotFound(name.to_string()))
    }

    pub fn collection_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.cols.lock().unwrap().keys().cloned().collect();
        v.sort();
        v
    }

    pub fn checkpoint_all(&self) -> Result<()> {
        for c in self.cols.lock().unwrap().values() {
            c.checkpoint()?;
        }
        Ok(())
    }

    pub fn checkpoint_all_keep_wal(&self) -> Result<()> {
        for c in self.cols.lock().unwrap().values() {
            c.checkpoint_keep_wal()?;
        }
        Ok(())
    }

    /// Consistent online backup: forces a checkpoint per collection (a clean
    /// cut including pending WAL ops), then copies the immutable listed
    /// segments + manifest into `dest`. Restore == copy back or simply
    /// `Db::open(dest)`. Blocks writers for the duration of the file copies.
    pub fn backup_to(&self, dest: impl AsRef<Path>) -> Result<PathBuf> {
        let dest = dest.as_ref();
        fs::create_dir_all(dest.join(COLLECTIONS_DIR))?;
        self.checkpoint_all()?;
        for name in self.collection_names() {
            let col = self.collection(&name)?;
            let dst_col = dest.join(COLLECTIONS_DIR).join(&name);
            fs::create_dir_all(&dst_col)?;
            for entry in fs::read_dir(col.dir_path())? {
                let entry = entry?;
                let fname = entry.file_name().to_string_lossy().to_string();
                if fname == MANIFEST_FILE || (fname.starts_with("seg-") && fname.ends_with(".jsonl"))
                {
                    fs::copy(entry.path(), dst_col.join(&fname))?;
                }
            }
        }
        Ok(dest.to_path_buf())
    }

    /// Restore a backup produced by `backup_to`: copies files into `dest`
    /// then opens it. The source directory is left untouched.
    pub fn restore_from(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> Result<Db> {
        let (src, dest) = (src.as_ref(), dest.as_ref());
        if dest.exists() {
            return Err(Error::invalid("restore destination already exists"));
        }
        fs::create_dir_all(dest)?;
        copy_dir_recursive(src, dest)?;
        Db::open(dest)
    }

    /// Export all collections to newline-delimited JSON (ENGR-1 rows wrapped
    /// with their collection name). Sketches are included so imports are
    /// bit-faithful without re-encoding.
    pub fn export_jsonl<W: std::io::Write>(&self, mut w: W) -> Result<u64> {
        #[derive(serde::Serialize)]
        struct ExportRow<'a> {
            collection: &'a str,
            row: &'a crate::storage::SnapshotRow,
        }
        let mut count = 0u64;
        for name in self.collection_names() {
            let col = self.collection(&name)?;
            col.checkpoint()?;
            let m = crate::storage::read_manifest(col.dir_path())?;
            for snap in &m.snapshots {
                for row in crate::storage::read_snapshot(col.dir_path(), snap)? {
                    serde_json::to_writer(&mut w, &ExportRow { collection: &name, row: &row })?;
                    w.write_all(b"\n")?;
                    count += 1;
                }
            }
        }
        w.flush()?;
        Ok(count)
    }

    /// Import rows written by `export_jsonl`. Idempotent on id collisions.
    pub fn import_jsonl<R: std::io::Read>(&self, mut r: R) -> Result<(u64, u64)> {
        #[derive(serde::Deserialize)]
        struct ExportRow {
            collection: String,
            row: crate::storage::SnapshotRow,
        }
        let mut buf = String::new();
        r.read_to_string(&mut buf)?;
        let mut imported = 0u64;
        let mut skipped = 0u64;
        for line in buf.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let er: ExportRow = serde_json::from_str(line)
                .map_err(|e| Error::invalid(format!("bad export row: {e}")))?;
            if !valid_name(&er.collection) {
                skipped += 1;
                continue;
            }
            let col = self.create_collection(&er.collection)?;
            match col.import_row(er.row)? {
                true => imported += 1,
                false => skipped += 1,
            }
        }
        Ok((imported, skipped))
    }

    /// Structural integrity walk over every persisted row.
    pub fn verify(&self) -> VerifyReport {
        let mut report = VerifyReport::default();
        for name in self.collection_names() {
            report.collections += 1;
            let Ok(col) = self.collection(&name) else {
                continue;
            };
            let dir = col.dir_path();
            let Ok(m) = crate::storage::read_manifest(dir) else {
                report
                    .errors
                    .push(format!("{name}: manifest unreadable"));
                continue;
            };
            let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
            for snap in &m.snapshots {
                match crate::storage::read_snapshot(dir, snap) {
                    Err(e) => report.errors.push(format!("{name}/{snap}: {e}")),
                    Ok(rows) => {
                        for row in rows {
                            report.rows += 1;
                            check_row(&name, &row, &mut seen, &mut report.errors);
                        }
                    }
                }
            }
        }
        report
    }

    /// Drop dead records older than `retention_secs` across all collections.
    /// Returns (live_dead_removed, archived_removed) totals.
    pub fn compact(&self, retention_secs: u64) -> Result<(usize, usize)> {
        let mut a = 0usize;
        let mut b = 0usize;
        for name in self.collection_names() {
            let col = self.collection(&name)?;
            let (x, y) = col.compact(retention_secs)?;
            a += x;
            b += y;
        }
        Ok((a, b))
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

#[derive(Default, Debug)]
pub struct VerifyReport {
    pub collections: usize,
    pub rows: u64,
    pub errors: Vec<String>,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }
}

fn check_row(
    col: &str,
    row: &SnapshotRow,
    seen: &mut std::collections::HashSet<u64>,
    errors: &mut Vec<String>,
) {
    let r = &row.record;
    if !seen.insert(r.id) {
        errors.push(format!("{col}: duplicate id {}", r.id));
    }
    if r.text.trim().is_empty() {
        errors.push(format!("{col}: empty text on id {}", r.id));
    }
    if !(0.0..=1.0).contains(&r.importance) {
        errors.push(format!("{col}: importance out of range on id {}", r.id));
    }
    if !row.sk.words.is_empty() && row.sk.words.len() != crate::sketch::WORDS {
        errors.push(format!("{col}: sketch word count on id {}", r.id));
    }
    if row.lex.idx.len() != row.lex.vals.len() {
        errors.push(format!("{col}: lex idx/vals mismatch on id {}", r.id));
    }
    if row.lex.idx.windows(2).any(|w| w[0] >= w[1]) {
        errors.push(format!("{col}: lex not sorted on id {}", r.id));
    }
    if let Some(v) = r.valid_to {
        if v < r.event_time {
            errors.push(format!("{col}: valid_to before event_time on id {}", r.id));
        }
    }
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &dest.join(&name))?;
        } else if name != LOCK_FILE {
            fs::copy(entry.path(), dest.join(&name))?;
        }
    }
    Ok(())
}

