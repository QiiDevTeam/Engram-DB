# EngramDB

A vector database designed for AI long-term memory and long-context retrieval. No external models required out of the box (built-in HashNgram deterministic encoder), with external embedding models available only as optional codecs.

## Repository Structure

```
db/                 Database core (Rust engine)
  src/              Engine: encoder/index (HNSW·Vamana·IVF-PQ)/storage/WAL/compactor/association graph
  src/bin/          engram-server (local HTTP/JSON service)
  src/ffi.rs        Stable C ABI (feature "capi", enabled by default)
  include/engram.h  C header file (single source of truth)
  gpu/              CUDA kernels (NVRTC runtime compilation, RTX实测 shows 19× speedup for steady-state queries)
  examples/         quickstart / eval / bench_indexes / vamana_tune
sdks/
  rust/             Rust SDK crate `engram` (facade + prelude)
  python/           Python package `engramdb` (PyO3, maturin build)
  cpp/              C++ SDK (header-only RAII + CMake)
go/                 Go package (local server client)
docs/ENGR-1.md      Disk format specification v1
BENCH.md            Performance and recall benchmark report
```

## Quick Start

### Rust Package

```rust
use engram::prelude::*;

let db = engram::open("./memory")?;
let col = db.create_collection("assistant")?;
col.remember_about("User prefers dark theme", "user.theme", 0.9)?;
let block = col.recall_for_prompt("theme preference", 300)?;
let hits = col.recall(RecallOpts::new("theme preference").budget_tokens(300))?;
```

```bash
cargo test --workspace
cargo run -p engram-db --example quickstart
```

### Python Package

```bash
cd sdks/python && pip install .        # maturin build; or cargo build -p engram-python then copy pyd
```

```python
import engram
db = engram.Db("./memory")
col = db.create_collection("assistant")
col.remember("User lives in Shanghai", subject="user.city")
hits = col.recall("Where does the user live?", budget_tokens=200)
print(hits[0].score, hits[0].text, hits[0].tier)
col.consolidate(min_cluster=3)          # Consolidation: cold data → Summary + archive
db.checkpoint_all()
```

### C++ Package

```bash
cargo build -p engram-db               # Produces engram_db.dll/.so
cmake -S sdks/cpp -B sdks/cpp/build && cmake --build sdks/cpp/build
```

### Go Package

```bash
cargo run -p engram-db --bin engram-server -- --path ./data --addr 127.0.0.1:9379
cd go && ENGRAM_TEST_URL=http://127.0.0.1:9379 go test ./...
```

## Core API (isomorphic across languages)

| Method | Description |
|---|---|
| remember | Write a memory; optional subject/tags/importance/event_time |
| recall(budget_tokens) | Token-budgeted recall: fused scoring → MMR → budget truncation; profiles: chat/agent-task/overview |
| forget / forget_subject | Soft deletion (bitemporal version chain, supports as_of lookback) |
| consolidate | Compactor: cold data clustering → Summary → child node archiving |
| backup_to / restore_from / export_jsonl / import_jsonl / verify / compact | Backup and restore suite |

## Roadmap

- [x] M0-M4 See BENCH.md for smoke test records
- [x] M5 Full migration of Go-vse index + AVX2 SIMD layer + SketchHnsw graph hot path
- [x] M6 Local data safety: cross-process lock/WAL tail tolerance/backup restore/export import/verify/compact/PITR archive
- [ ] Future: Summary multi-layer tree, external embedding codec registry, SketchHnsw insertion speedup
