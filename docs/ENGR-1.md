# ENGR-1 磁盘格式规范（v1）

任何语言实现本规范即可与 EngramDB 数据目录互操作。

## 目录布局

```
<root>/                       # Db::open(root) 的根目录
  collections/
    <name>/                   # 一个 collection 一个子目录
      manifest.json           # 原子更新的清单（写 tmp 后 rename）
      seg-000001.jsonl        # 快照段（JSON Lines，每行一个 SnapshotRow）
      wal.jsonl               # 预写日志（JSON Lines，每行一个 WalOp）
```

collection 名约束：`[A-Za-z0-9_-]{1,64}`。

## manifest.json

```json
{
  "version": 1,
  "next_id": 42,
  "snapshots": ["seg-000003.jsonl"],
  "df": { "12345678901234567": 17 },
  "wal_archives": ["wal-a000004-1730000000.archived"]
}
```

- `version`: 恒为 1。
- `next_id`: 下一次 remember 分配的记录 id，加载时取 max(next_id, 最大已见 id+1)。
- `snapshots`: 按序加载的段文件列表；未列出的 seg-*.jsonl 是垃圾，可删除。
- `df`: 词特征哈希(u64 十进制字符串) → 文档频率，用于 IDF。可选（缺省视为空）。
- `wal_archives`: 归档 WAL 段（checkpoint_keep_wal 产生，不截断），恢复时在
  快照之后、活动 WAL 之前按序重放。PITR 基建。

## 锁文件

`<root>/engram.lock`：Db::open 时获取跨进程独占锁（Windows 共享模式 0 句柄 /
Unix flock LOCK_EX|LOCK_NB），第二个进程打开同一目录报 `Error::Locked`。
关闭句柄即释放。

## SnapshotRow（seg-*.jsonl 每行）

```json
{"record": {...}, "lex": {"idx": [..], "vals": [..]}, "sk": {"words": [..], "nz": 192}, "archived": false}
```

record 字段：`id, text, subject(null|str), tags[], event_time, ingest_time,
valid_to(null|int), importance(f32), hits, last_hit(null|int),
source_ids[]`（后两者 serde default，兼容旧文件）。`archived=true` 的行加载进归档区，
不参与默认召回。

## WalOp（wal.jsonl 每行，tag 字段 "op"）

| op | 字段 | 语义 |
|---|---|---|
| `remember` | `row: SnapshotRow` | 插入/覆盖记录 |
| `forget` | `id, at` | 软失效：valid_to=at |
| `touch` | `id, at` | 访问强化：hits+=1, last_hit=at |
| `hard_delete` | `id` | 从任一区移除 |
| `archive` | `id` | live→archive 区移动 |

## 崩溃恢复顺序

1. 读 manifest，按 snapshots 列表顺序加载行（route by archived）。
2. 按 WAL 顺序重放全部 op。
3. 打开 WAL 进入追加模式；checkpoint 时：写新段 → fsync → 原子替换
   manifest → 清理未列出的旧段 → 截断重建 WAL。

## 编码器约定

内置 HashNgram-1024：DIM=1024；三值 sketch 每维 2-bit（0→00, +1→01, -1→10），
小端打包进 u64 words；相似度 = 匹配符号积 / sqrt(nz_a·nz_b)。
跨实现必须字节级一致（splitmix64 + FNV 混合的特征哈希，见 encoder.rs）。
