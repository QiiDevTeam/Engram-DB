#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;

use crate::collection::{Collection as CoreCollection, RecallOpts};
use crate::db::Db as CoreDb;
use crate::{Profile, RememberOpts};

thread_local! {
    static LAST_ERR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_err(msg: String) {
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERR.with(|e| *e.borrow_mut() = Some(c));
}

fn take_err() -> Option<CString> {
    LAST_ERR.with(|e| e.borrow_mut().take())
}

const ENGRAM_OK: i32 = 0;
const ENGRAM_ERR: i32 = -1;
const ENGRAM_ERR_NULL: i32 = -2;

pub struct CDb {
    inner: CoreDb,
}

pub struct CCol {
    inner: Arc<CoreCollection>,
}

#[repr(C)]
pub struct CHit {
    pub id: u64,
    pub score: f32,
    pub est_tokens: u32,
    pub text: *mut c_char,
}

unsafe fn cstr<'a>(p: *const c_char) -> Result<&'a str, String> {
    if p.is_null() {
        return Err("null string pointer".into());
    }
    CStr::from_ptr(p)
        .to_str()
        .map_err(|e| format!("invalid utf-8: {e}"))
}

#[no_mangle]
pub extern "C" fn engram_version() -> *const c_char {
    concat!("engram ", env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn engram_last_error(buf: *mut c_char, cap: usize) -> i32 {
    match take_err() {
        Some(msg) if !buf.is_null() && cap > 0 => {
            let bytes = msg.as_bytes_with_nul();
            let n = bytes.len().min(cap);
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, n);
            }
            n as i32
        }
        Some(_) => 0,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn engram_open(path: *const c_char) -> *mut CDb {
    let r = unsafe { cstr(path) }
        .and_then(|p| CoreDb::open(p).map_err(|e| e.to_string()));
    match r {
        Ok(db) => Box::into_raw(Box::new(CDb { inner: db })),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// handle must come from engram_open and not be closed twice.
#[no_mangle]
pub unsafe extern "C" fn engram_close(db: *mut CDb) {
    if !db.is_null() {
        drop(Box::from_raw(db));
    }
}

#[no_mangle]
pub extern "C" fn engram_collection(db: *mut CDb, name: *const c_char) -> *mut CCol {
    if db.is_null() {
        set_err("null db handle".into());
        return std::ptr::null_mut();
    }
    let r = unsafe { cstr(name) }
        .and_then(|name| unsafe { &*db }.inner.create_collection(name).map_err(|e| e.to_string()));
    match r {
        Ok(col) => Box::into_raw(Box::new(CCol { inner: col })),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// handle must come from engram_collection and not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn engram_collection_close(col: *mut CCol) {
    if !col.is_null() {
        drop(Box::from_raw(col));
    }
}

#[no_mangle]
pub extern "C" fn engram_remember(
    col: *const CCol,
    text: *const c_char,
    subject: *const c_char,
    importance: f32,
    event_time: i64,
) -> i64 {
    if col.is_null() {
        set_err("null collection handle".into());
        return -1;
    }
    let r = (|| -> Result<u64, String> {
        let text = unsafe { cstr(text) }?;
        let mut opts = RememberOpts::new(text).importance(importance);
        if !subject.is_null() {
            opts = opts.subject(unsafe { cstr(subject) }?);
        }
        if event_time > 0 {
            opts = opts.event_time(event_time);
        }
        unsafe { &*col }.inner.remember(opts).map_err(|e| e.to_string())
    })();
    match r {
        Ok(id) => id as i64,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn engram_recall(
    col: *const CCol,
    query: *const c_char,
    budget_tokens: usize,
    k_max: usize,
    profile: u8,
    include_cold: i32,
    out_hits: *mut *mut CHit,
    out_count: *mut usize,
) -> i32 {
    if col.is_null() || out_hits.is_null() || out_count.is_null() {
        return ENGRAM_ERR_NULL;
    }
    let r = (|| -> Result<Vec<crate::collection::Hit>, String> {
        let query = unsafe { cstr(query) }?;
        let p = match profile {
            1 => Profile::AgentTask,
            2 => Profile::Overview,
            _ => Profile::Chat,
        };
        let opts = RecallOpts::new(query)
            .budget_tokens(budget_tokens)
            .k_max(k_max)
            .profile(p)
            .include_cold(include_cold != 0);
        unsafe { &*col }.inner.recall(opts).map_err(|e| e.to_string())
    })();
    match r {
        Ok(hits) => {
            let mut arr: Vec<CHit> = hits
                .into_iter()
                .map(|h| CHit {
                    id: h.id,
                    score: h.score,
                    est_tokens: h.estimated_tokens as u32,
                    text: CString::new(h.text)
                        .unwrap_or_default()
                        .into_raw(),
                })
                .collect();
            let count = arr.len();
            let ptr = arr.as_mut_ptr();
            std::mem::forget(arr);
            unsafe {
                *out_hits = ptr;
                *out_count = count;
            }
            ENGRAM_OK
        }
        Err(e) => {
            set_err(e);
            ENGRAM_ERR
        }
    }
}

/// # Safety
/// hits/count must come from a successful engram_recall call.
#[no_mangle]
pub unsafe extern "C" fn engram_free_hits(hits: *mut CHit, count: usize) {
    if hits.is_null() || count == 0 {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(hits, count);
    for h in slice.iter() {
        if !h.text.is_null() {
            drop(CString::from_raw(h.text));
        }
    }
    drop(Box::from_raw(slice as *mut [CHit]));
}

#[no_mangle]
pub extern "C" fn engram_forget(col: *const CCol, id: u64) -> i32 {
    if col.is_null() {
        return ENGRAM_ERR_NULL;
    }
    match unsafe { &*col }.inner.forget(id) {
        Ok(()) => ENGRAM_OK,
        Err(e) => {
            set_err(e.to_string());
            ENGRAM_ERR
        }
    }
}

#[no_mangle]
pub extern "C" fn engram_checkpoint(db: *mut CDb) -> i32 {
    if db.is_null() {
        return ENGRAM_ERR_NULL;
    }
    match unsafe { &*db }.inner.checkpoint_all() {
        Ok(()) => ENGRAM_OK,
        Err(e) => {
            set_err(e.to_string());
            ENGRAM_ERR
        }
    }
}



/// # Safety
/// Frees a string previously returned by engram_verify.
#[no_mangle]
pub unsafe extern "C" fn engram_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

#[no_mangle]
pub extern "C" fn engram_forget_subject(col: *const CCol, subject: *const c_char) -> i64 {
    if col.is_null() {
        set_err("null collection handle".into());
        return -1;
    }
    let r = unsafe { cstr(subject) }
        .and_then(|s| unsafe { &*col }.inner.forget_subject(s).map_err(|e| e.to_string()));
    match r {
        Ok(n) => n as i64,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// # Safety
/// col must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn engram_hard_delete(col: *const CCol, id: u64) -> i32 {
    if col.is_null() {
        return ENGRAM_ERR_NULL;
    }
    match (*col).inner.hard_delete(id) {
        Ok(()) => ENGRAM_OK,
        Err(e) => {
            set_err(e.to_string());
            ENGRAM_ERR
        }
    }
}

#[no_mangle]
pub extern "C" fn engram_stats(
    col: *const CCol,
    out_live: *mut usize,
    out_total: *mut usize,
) -> i32 {
    if col.is_null() || out_live.is_null() || out_total.is_null() {
        return ENGRAM_ERR_NULL;
    }
    let (live, total) = unsafe { &*col }.inner.stats();
    unsafe {
        *out_live = live;
        *out_total = total;
    }
    ENGRAM_OK
}

#[no_mangle]
pub extern "C" fn engram_consolidate(
    col: *const CCol,
    min_cluster: usize,
    out_clusters: *mut usize,
    out_archived: *mut usize,
    out_summaries: *mut usize,
) -> i32 {
    if col.is_null() || out_clusters.is_null() || out_archived.is_null() || out_summaries.is_null()
    {
        return ENGRAM_ERR_NULL;
    }
    let cfg = crate::collection::ConsolidateCfg::new().min_cluster(min_cluster);
    match unsafe { (*col).inner.consolidate(cfg) } {
        Ok(r) => unsafe {
            *out_clusters = r.clusters;
            *out_archived = r.archived;
            *out_summaries = r.summaries_created;
        },
        Err(e) => {
            set_err(e.to_string());
            return ENGRAM_ERR;
        }
    }
    ENGRAM_OK
}

/// # Safety
/// db must come from engram_open.
#[no_mangle]
pub unsafe extern "C" fn engram_checkpoint_keep_wal(db: *mut CDb) -> i32 {
    if db.is_null() {
        return ENGRAM_ERR_NULL;
    }
    match (*db).inner.checkpoint_all_keep_wal() {
        Ok(()) => ENGRAM_OK,
        Err(e) => {
            set_err(e.to_string());
            ENGRAM_ERR
        }
    }
}

#[no_mangle]
pub extern "C" fn engram_compact(
    db: *mut CDb,
    retention_secs: u64,
    out_live_dead: *mut usize,
    out_archived: *mut usize,
) -> i32 {
    if db.is_null() || out_live_dead.is_null() || out_archived.is_null() {
        return ENGRAM_ERR_NULL;
    }
    match unsafe { (*db).inner.compact(retention_secs) } {
        Ok((a, b)) => unsafe {
            *out_live_dead = a;
            *out_archived = b;
        },
        Err(e) => {
            set_err(e.to_string());
            return ENGRAM_ERR;
        }
    }
    ENGRAM_OK
}

#[no_mangle]
pub extern "C" fn engram_backup(db: *mut CDb, dest: *const c_char) -> i32 {
    if db.is_null() {
        return ENGRAM_ERR_NULL;
    }
    let r = unsafe { cstr(dest) }
        .and_then(|d| unsafe { (*db).inner.backup_to(d) }.map(|_| ()).map_err(|e| e.to_string()));
    match r {
        Ok(()) => ENGRAM_OK,
        Err(e) => {
            set_err(e);
            ENGRAM_ERR
        }
    }
}

#[no_mangle]
pub extern "C" fn engram_export_jsonl(db: *mut CDb, path: *const c_char, out_count: *mut u64) -> i32 {
    if db.is_null() || path.is_null() || out_count.is_null() {
        return ENGRAM_ERR_NULL;
    }
    let r = (|| -> Result<u64, String> {
        let p = unsafe { cstr(path) }?;
        let f = std::fs::File::create(p).map_err(|e| e.to_string())?;
        unsafe { (*db).inner.export_jsonl(f) }.map_err(|e| e.to_string())
    })();
    match r {
        Ok(n) => unsafe {
            *out_count = n;
        },
        Err(e) => {
            set_err(e);
            return ENGRAM_ERR;
        }
    }
    ENGRAM_OK
}

#[no_mangle]
pub extern "C" fn engram_import_jsonl(
    db: *mut CDb,
    path: *const c_char,
    out_imported: *mut u64,
    out_skipped: *mut u64,
) -> i32 {
    if db.is_null() || path.is_null() || out_imported.is_null() || out_skipped.is_null() {
        return ENGRAM_ERR_NULL;
    }
    let r = (|| -> Result<(u64, u64), String> {
        let p = unsafe { cstr(path) }?;
        let f = std::fs::File::open(p).map_err(|e| e.to_string())?;
        unsafe { (*db).inner.import_jsonl(f) }.map_err(|e| e.to_string())
    })();
    match r {
        Ok((i, s)) => unsafe {
            *out_imported = i;
            *out_skipped = s;
        },
        Err(e) => {
            set_err(e);
            return ENGRAM_ERR;
        }
    }
    ENGRAM_OK
}

/// Verify returns a JSON document:
/// {"collections":N,"rows":N,"ok":bool,"errors":["..."]}
#[no_mangle]
pub extern "C" fn engram_verify(db: *mut CDb, out_json: *mut *mut c_char) -> i32 {
    if db.is_null() || out_json.is_null() {
        return ENGRAM_ERR_NULL;
    }
    let report = unsafe { &*db }.inner.verify();
    let errors: Vec<&str> = report.errors.iter().map(|s| s.as_str()).collect();
    let json = serde_json::json!({
        "collections": report.collections,
        "rows": report.rows,
        "ok": report.ok(),
        "errors": errors,
    });
    let s = match serde_json::to_string(&json) {
        Ok(s) => s,
        Err(e) => {
            set_err(e.to_string());
            return ENGRAM_ERR;
        }
    };
    match CString::new(s) {
        Ok(c) => unsafe {
            *out_json = c.into_raw();
        },
        Err(_) => return ENGRAM_ERR,
    }
    ENGRAM_OK
}


