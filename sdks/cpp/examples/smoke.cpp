// EngramDB C++ SDK smoke: exercises the header-only RAII wrapper end-to-end.
// Build (repo root):
//   cargo build -p engram-db
//   g++ -std=c++17 -O1 -I db/include -I sdks/cpp/include \
//       sdks/cpp/examples/smoke.cpp -o smoke -L target/debug -lengram_db

#include <engram.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    std::string dir = std::string(std::getenv("TEMP") ? std::getenv("TEMP") : ".") +
                      "/engram-cpp-sdk-smoke";

    try {
        auto db = engram::open_database(dir);
        auto col = db.collection("main");

        col.remember("user prefers rust over go for systems work", "", 0.8f);
        col.remember("postgres replication lag incident last friday");

        db.checkpoint();
        db.checkpoint_keep_wal();

        auto compacted = db.compact(/*retention_secs*/ 0);
        printf("compact: live_dead=%zu archived=%zu\n",
               compacted.live_dead_removed, compacted.archived_removed);

        auto verified = db.verify();
        printf("verify: rows=%zu ok=%s\n", verified.rows,
               verified.ok ? "true" : "false");
        if (!verified.ok) throw engram::Error("verify reported errors");

        auto hits = col.recall("which language does user prefer", 300, 16);
        if (hits.empty() || hits[0].text.find("rust") == std::string::npos) {
            fprintf(stderr, "unexpected top hit\n");
            return 2;
        }
        printf("top: [%u tok] %s\n", hits[0].estimated_tokens,
               hits[0].text.c_str());

        auto stats = col.stats();
        printf("stats: live=%zu total=%zu\n", stats.live, stats.total);

        auto report = col.consolidate(/*min_cluster*/ 3);
        printf("consolidate: clusters=%zu archived=%zu summaries=%zu\n",
               report.clusters, report.archived, report.summaries_created);

        uint64_t exported = db.export_jsonl(dir + "/export.jsonl");
        auto re = db.import_jsonl(dir + "/export.jsonl");
        printf("export/import: %llu -> imported=%llu skipped=%llu\n",
               static_cast<unsigned long long>(exported),
               static_cast<unsigned long long>(re.first),
               static_cast<unsigned long long>(re.second));

        printf("C++ SDK smoke OK\n");
        return 0;
    } catch (const engram::Error& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 3;
    }
}
