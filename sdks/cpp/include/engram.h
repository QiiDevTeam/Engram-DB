#ifndef ENGRAM_H
#define ENGRAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ENGRAM_OK 0
#define ENGRAM_ERR (-1)
#define ENGRAM_ERR_NULL (-2)

typedef struct CDb CDb;
typedef struct CCol CCol;

typedef struct CHit {
    uint64_t id;
    float score;
    uint32_t est_tokens;
    char *text;
} CHit;

const char *engram_version(void);
int32_t engram_last_error(char *buf, size_t cap);

CDb *engram_open(const char *path);
void engram_close(CDb *db);

CCol *engram_collection(CDb *db, const char *name);
void engram_collection_close(CCol *col);

int64_t engram_remember(CCol *col, const char *text, const char *subject,
                        float importance, int64_t event_time);

int32_t engram_recall(CCol *col, const char *query, size_t budget_tokens,
                      size_t k_max, uint8_t profile, int32_t include_cold,
                      CHit **out_hits, size_t *out_count);
void engram_free_hits(CHit *hits, size_t count);

int32_t engram_forget(CCol *col, uint64_t id);
int32_t engram_checkpoint(CDb *db);

/* ---- maintenance / high-level operations ---- */

void engram_free_string(char *s);

int64_t engram_forget_subject(CCol *col, const char *subject);
int32_t engram_hard_delete(CCol *col, uint64_t id);
int32_t engram_stats(CCol *col, size_t *out_live, size_t *out_total);

int32_t engram_consolidate(CCol *col, size_t min_cluster,
                           size_t *out_clusters, size_t *out_archived,
                           size_t *out_summaries);

int32_t engram_checkpoint_keep_wal(CDb *db);
int32_t engram_compact(CDb *db, uint64_t retention_secs,
                       size_t *out_live_dead, size_t *out_archived);
int32_t engram_backup(CDb *db, const char *dest);

int32_t engram_export_jsonl(CDb *db, const char *path, uint64_t *out_count);
int32_t engram_import_jsonl(CDb *db, const char *path,
                            uint64_t *out_imported, uint64_t *out_skipped);
int32_t engram_verify(CDb *db, char **out_json);

#ifdef __cplusplus
}
#endif

#endif
