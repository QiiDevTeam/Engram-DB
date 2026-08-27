pub mod collection;
pub mod db;
pub mod encoder;
pub mod error;
#[cfg(feature = "capi")]
pub mod ffi;
pub mod graph;
pub mod gpu;
pub mod index;
pub mod recall;
pub mod salience;
pub mod server;
pub mod simd;
pub mod sketch;
pub mod sketch_hnsw;
pub mod sparse;
pub mod storage;
pub mod tokenizer;
pub mod types;

pub use collection::{Collection, Hit, RecallOpts};
pub use db::Db;
pub use error::{Error, Result};
pub use recall::Profile;
pub use types::{Record, RememberOpts};

pub fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}


