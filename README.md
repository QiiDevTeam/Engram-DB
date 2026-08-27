# EngramDB

为 AI 长期记忆与长上下文检索设计的向量数据库。零外挂模型可用（内置 HashNgram
确定性编码器），外部 embedding 模型仅作为可选 codec。

## 仓库结构

```
db/                 数据库本体（Rust 引擎）
  src/              引擎：编码器/索引(HNSW·Vamana·IVF-PQ)/存储/WAL/巩固器/关联图
  src/bin/          engram-server（本地 HTTP/JSON 服务）
  src/ffi.rs        稳定 C ABI（feature "capi"，默认开启）
  include/engram.h  C 头文件（唯一真源）
  gpu/              CUDA kernel（NVRTC 运行时编译，RTX 实测常驻查询 19× 加速）
  examples/         quickstart / eval / bench_indexes / vamana_tune
sdks/
  rust/             Rust SDK crate `engram`（门面 + prelude）
  python/           Python 包 `engramdb`（PyO3，maturin 构建）
  cpp/              C++ SDK（header-only RAII + CMake）
go/                 Go 包（本地 server 客户端）
docs/ENGR-1.md      磁盘格式规范 v1
BENCH.md            性能与召回评测报告
```

## 快速开始

### Rust 包

```rust
use engram::prelude::*;

let db = engram::open("./memory")?;
let col = db.create_collection("assistant")?;
col.remember_about("用户偏好深色主题", "user.theme", 0.9)?;`nlet block = col.recall_for_prompt("主题偏好", 300)?;
let hits = col.recall(RecallOpts::new("主题偏好").budget_tokens(300))?;
```

```bash
cargo test --workspace
cargo run -p engram-db --example quickstart
```

### Python 包

```bash
cd sdks/python && pip install .        # maturin 构建；或 cargo build -p engram-python 后拷贝 pyd
```

```python
import engram
db = engram.Db("./memory")
col = db.create_collection("assistant")
col.remember("用户住在上海", subject="user.city")
hits = col.recall("用户住在哪里", budget_tokens=200)
print(hits[0].score, hits[0].text, hits[0].tier)
col.consolidate(min_cluster=3)          # 巩固：冷数据→Summary+归档
db.checkpoint_all()
```

### C++ 包

```bash
cargo build -p engram-db               # 产出 engram_db.dll/.so
cmake -S sdks/cpp -B sdks/cpp/build && cmake --build sdks/cpp/build
```

### Go 包

```bash
cargo run -p engram-db --bin engram-server -- --path ./data --addr 127.0.0.1:9379
cd go && ENGRAM_TEST_URL=http://127.0.0.1:9379 go test ./...
```

## 核心 API（各语言同构）

| 方法 | 说明 |
|---|---|
| remember | 写入记忆；可选 subject/tags/importance/event_time |
| recall(budget_tokens) | token 预算召回：融合打分→MMR→预算截断；profile: chat/agent-task/overview |
| forget / forget_subject | 软遗忘（双时态版本链，支持 as_of 回溯） |
| consolidate | 巩固器：冷数据聚类→Summary→子节点归档 |
| backup_to / restore_from / export_jsonl / import_jsonl / verify / compact | 备份恢复套件 |

## Roadmap

- [x] M0-M4 见 BENCH.md 冒烟记录
- [x] M5 Go-vse 索引全量移植 + AVX2 SIMD 层 + SketchHnsw 图热路径
- [x] M6 本地数据安全：跨进程锁/WAL 断尾容忍/备份恢复/导出导入/verify/compact/PITR 归档
- [ ] 后续：Summary 多层树、外部 embedding codec 注册表、SketchHnsw 插入提速


