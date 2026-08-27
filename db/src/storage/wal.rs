use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::storage::SnapshotRow;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalOp {
    Remember { row: SnapshotRow },
    Forget { id: u64, at: i64 },
    Touch { id: u64, at: i64 },
    HardDelete { id: u64 },
    Archive { id: u64 },
}

pub struct WalWriter {
    inner: BufWriter<File>,
}

impl WalWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        Ok(WalWriter {
            inner: BufWriter::new(f),
        })
    }

    pub fn open_append(path: &Path) -> Result<Self> {
        let f = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(WalWriter {
            inner: BufWriter::new(f),
        })
    }

    pub fn append(&mut self, op: &WalOp) -> Result<()> {
        serde_json::to_writer(&mut self.inner, op)?;
        self.inner.write_all(b"\n")?;
        self.inner.flush()?;
        Ok(())
    }

    pub fn sync(&mut self) -> Result<()> {
        self.inner.flush()?;
        self.inner.get_ref().sync_all()?;
        Ok(())
    }
}

/// Replay WAL applying ops in order.
///
/// Crash-tolerance: the LAST line of a WAL can be torn (partial write when
/// the process died mid-append). We therefore STOP at the first unparsable
/// line and report it as `skipped` instead of failing the whole open — data
/// after an unknown-corruption point is treated as suspect by design.
pub fn replay(path: &Path, mut apply: impl FnMut(WalOp)) -> Result<(usize, usize)> {
    if !path.exists() {
        return Ok((0, 0));
    }
    let f = BufReader::new(File::open(path)?);
    let mut n = 0usize;
    let mut skipped = 0usize;
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<WalOp>(&line) {
            Ok(op) => {
                apply(op);
                n += 1;
            }
            Err(_) => {
                skipped += 1;
                break;
            }
        }
    }
    Ok((n, skipped))
}

