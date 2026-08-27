#pragma once
// EngramDB C++ SDK: header-only RAII wrapper over the stable C ABI
// (db/include/engram.h, exported by the engram_db cdylib).
//
// Linking: consume the import lib generated with the DLL, or use dynamic
// loading (see examples/smoke.cpp for a LoadLibrary-based pattern that needs
// no import library at all).

#include "engram.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engram {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

inline void check(int32_t rc) {
    if (rc != ENGRAM_OK) {
        char buf[512] = {0};
        int32_t n = engram_last_error(buf, sizeof buf);
        std::string msg = n > 0 ? std::string(buf, static_cast<size_t>(n - 1))
                                : "unknown engram error";
        throw Error(msg);
    }
}

struct Hit {
    uint64_t id;
    float score;
    uint32_t estimated_tokens;
    std::string text;
};

struct Stats {
    size_t live = 0;
    size_t total = 0;
};

struct ConsolidateReport {
    size_t clusters = 0;
    size_t archived = 0;
    size_t summaries_created = 0;
};

struct CompactReport {
    size_t live_dead_removed = 0;
    size_t archived_removed = 0;
};

struct VerifyReport {
    size_t collections = 0;
    size_t rows = 0;
    bool ok = false;
    std::vector<std::string> errors;

    // Minimal extractor for the fixed-shape JSON returned by engram_verify
    // (avoids pulling a JSON dependency into the header-only SDK).
    static VerifyReport parse(const std::string& json) {
        VerifyReport r;
        auto find_num = [&](const char* key) -> size_t {
            std::string k = "\"" + std::string(key) + "\":";
            auto p = json.find(k);
            if (p == std::string::npos) return 0;
            return static_cast<size_t>(std::strtoull(json.c_str() + p + k.size(), nullptr, 10));
        };
        r.collections = find_num("collections");
        r.rows = find_num("rows");
        r.ok = json.find("\"ok\":true") != std::string::npos;
        return r;
    }
};

namespace detail {
struct DbDeleter {
    void operator()(CDb* p) const noexcept {
        if (p) engram_close(p);
    }
};
struct ColDeleter {
    void operator()(CCol* p) const noexcept {
        if (p) engram_collection_close(p);
    }
};
} // namespace detail

class Collection;

class Db {
public:
    explicit Db(const std::string& path)
        : handle_(engram_open(path.c_str())) {
        if (!handle_) throw Error("engram_open failed");
    }
    Db(Db&&) = default;
    Db& operator=(Db&&) = default;
    Db(const Db&) = delete;
    Db& operator=(const Db&) = delete;

    Collection collection(const std::string& name);
    void checkpoint() { check(engram_checkpoint(handle_.get())); }

    /// PITR-friendly checkpoint (archives WAL instead of truncating).
    void checkpoint_keep_wal() { check(engram_checkpoint_keep_wal(handle_.get())); }

    /// Consistent online backup into `dest` (created if missing).
    std::string backup_to(const std::string& dest) {
        check(engram_backup(handle_.get(), dest.c_str()));
        return dest;
    }

    /// Drop dead records older than retention_secs.
    CompactReport compact(uint64_t retention_secs) {
        CompactReport r;
        check(engram_compact(handle_.get(), retention_secs,
                             &r.live_dead_removed, &r.archived_removed));
        return r;
    }

    /// Export all collections as NDJSON (ENGR-1 rows). Returns row count.
    uint64_t export_jsonl(const std::string& path) {
        uint64_t n = 0;
        check(engram_export_jsonl(handle_.get(), path.c_str(), &n));
        return n;
    }

    /// Import NDJSON rows; idempotent on ids. Returns (imported, skipped).
    std::pair<uint64_t, uint64_t> import_jsonl(const std::string& path) {
        uint64_t imported = 0, skipped = 0;
        check(engram_import_jsonl(handle_.get(), path.c_str(), &imported, &skipped));
        return {imported, skipped};
    }

    VerifyReport verify() {
        char* json = nullptr;
        check(engram_verify(handle_.get(), &json));
        std::string s = json ? json : "";
        engram_free_string(json);
        return VerifyReport::parse(s);
    }

private:
    friend class Collection;
    std::unique_ptr<CDb, detail::DbDeleter> handle_;
};

class Collection {
public:
    uint64_t remember(const std::string& text,
                      const std::string& subject = {},
                      float importance = 0.5f,
                      int64_t event_time = 0) {
        const char* subj = subject.empty() ? nullptr : subject.c_str();
        int64_t id = engram_remember(handle_.get(), text.c_str(), subj,
                                     importance, event_time);
        if (id < 0) throw Error("engram_remember failed");
        return static_cast<uint64_t>(id);
    }

    std::vector<Hit> recall(const std::string& query,
                            size_t budget_tokens = 512,
                            size_t k_max = 64,
                            uint8_t profile = 0,
                            bool include_cold = false) {
        CHit* hits = nullptr;
        size_t n = 0;
        check(engram_recall(handle_.get(), query.c_str(), budget_tokens,
                            k_max, profile,
                            include_cold ? 1 : 0, &hits, &n));
        std::vector<Hit> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            out.push_back(Hit{hits[i].id, hits[i].score,
                              hits[i].est_tokens,
                              hits[i].text ? hits[i].text : ""});
        }
        engram_free_hits(hits, n);
        return out;
    }

    void forget(uint64_t id) { check(engram_forget(handle_.get(), id)); }

    /// Soft-forget every live record with this subject; returns count.
    size_t forget_subject(const std::string& subject) {
        int64_t n = engram_forget_subject(handle_.get(), subject.c_str());
        if (n < 0) throw Error("engram_forget_subject failed");
        return static_cast<size_t>(n);
    }

    void hard_delete(uint64_t id) { check(engram_hard_delete(handle_.get(), id)); }

    Stats stats() {
        Stats s;
        check(engram_stats(handle_.get(), &s.live, &s.total));
        return s;
    }

    /// Cluster cold memories into extractive summaries and archive members.
    ConsolidateReport consolidate(size_t min_cluster = 3) {
        ConsolidateReport r;
        check(engram_consolidate(handle_.get(), min_cluster, &r.clusters,
                                 &r.archived, &r.summaries_created));
        return r;
    }

private:
    friend class Db;
    explicit Collection(CCol* p) : handle_(p) {}
    std::unique_ptr<CCol, detail::ColDeleter> handle_;
};

inline Collection Db::collection(const std::string& name) {
    CCol* col = engram_collection(handle_.get(), name.c_str());
    if (!col) throw Error("engram_collection failed");
    return Collection(col);
}

inline Db open_database(const std::string& path) { return Db(path); }

} // namespace engram
