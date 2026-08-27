pub mod lock;
pub mod wal;

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::sketch::Sketch;
use crate::sparse::SparseVec;
use crate::types::Record;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotRow {
    pub record: Record,
    pub lex: SparseVec,
    pub sk: Sketch,
    #[serde(default)]
    pub archived: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub next_id: u64,
    pub snapshots: Vec<String>,
    #[serde(default)]
    pub df: std::collections::HashMap<u64, u32>,
    /// Archived (non-truncated) WAL segments, oldest first — PITR building block.
    #[serde(default)]
    pub wal_archives: Vec<String>,
}

pub const MANIFEST_FILE: &str = "manifest.json";
pub const WAL_FILE: &str = "wal.jsonl";

pub fn snapshot_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}

pub fn write_manifest_atomic(dir: &Path, m: &Manifest) -> Result<()> {
    let tmp = dir.join(format!("{MANIFEST_FILE}.tmp"));
    {
        let f = File::create(&tmp)?;
        let mut w = BufWriter::new(f);
        serde_json::to_writer(&mut w, m)?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    fs::rename(&tmp, dir.join(MANIFEST_FILE))?;
    Ok(())
}

pub fn read_manifest(dir: &Path) -> Result<Manifest> {
    let p = dir.join(MANIFEST_FILE);
    if !p.exists() {
        return Ok(Manifest {
            version: 1,
            ..Default::default()
        });
    }
    let f = File::open(&p)?;
    let m = serde_json::from_reader(f)?;
    Ok(m)
}

pub fn write_snapshot(dir: &Path, name: &str, rows: impl Iterator<Item = SnapshotRow>) -> Result<()> {
    let p = dir.join(format!("{name}.tmp"));
    {
        let f = File::create(&p)?;
        let mut w = BufWriter::new(f);
        for row in rows {
            serde_json::to_writer(&mut w, &row)?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    fs::rename(&p, snapshot_path(dir, name))?;
    Ok(())
}

pub fn read_snapshot(dir: &Path, name: &str) -> Result<Vec<SnapshotRow>> {
    let p = snapshot_path(dir, name);
    if !p.exists() {
        return Err(Error::invalid(format!("snapshot missing: {name}")));
    }
    let f = BufReader::new(File::open(&p)?);
    let mut out = Vec::new();
    for line in f.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line)?);
    }
    Ok(out)
}

pub fn cleanup_unlisted_segments(dir: &Path, listed: &[String]) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(n) => n,
            None => continue,
        };
        if name.starts_with("seg-")
            && name.ends_with(".jsonl")
            && !listed.iter().any(|l| l == name)
        {
            let _ = fs::remove_file(entry.path());
        }
    }
    Ok(())
}

pub fn new_segment_name(seq: u64) -> String {
    format!("seg-{seq:06}.jsonl")
}

