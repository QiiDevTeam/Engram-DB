//! Cross-process exclusive lock on the database root.
//!
//! Prevents the classic local-tooling accident: an MCP server holding the DB
//! while a CLI/second agent process opens the same directory concurrently and
//! corrupts the snapshot/WAL pairing. RAII: dropping Db releases the lock.

use std::path::Path;

use crate::error::{Error, Result};

pub struct FileLock {
    inner: Inner,
}

#[cfg(windows)]
mod imp {
    use super::*;
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    pub(crate) struct Inner {
        handle: *mut std::ffi::c_void,
    }

    unsafe impl Send for Inner {}
    unsafe impl Sync for Inner {}

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateFileW(
            lpfilename: *const u16,
            dwdesiredaccess: u32,
            dwsharemode: u32,
            lpsecurityattributes: *mut std::ffi::c_void,
            dwcreationdisposition: u32,
            dwflagsandattributes: u32,
            htemplatefile: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void;
        fn CloseHandle(hobject: *mut std::ffi::c_void) -> i32;
    }

    const GENERIC_WRITE: u32 = 0x4000_0000;
    const OPEN_ALWAYS: u32 = 4;
    const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
    const INVALID_HANDLE_VALUE: *mut std::ffi::c_void = -1isize as *mut std::ffi::c_void;

    pub(crate) fn acquire(path: &Path) -> Result<Inner> {
        let wide: Vec<u16> = OsStr::new(path)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let handle = unsafe {
            CreateFileW(
                wide.as_ptr(),
                GENERIC_WRITE,
                0, // exclusive: second opener gets sharing violation
                std::ptr::null_mut(),
                OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE || handle.is_null() {
            return Err(Error::Locked(
                "database directory is locked by another process".into(),
            ));
        }
        Ok(Inner { handle })
    }

    impl Drop for Inner {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.handle);
            }
        }
    }
}

#[cfg(unix)]
mod imp {
    use super::*;
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    pub(crate) struct Inner {
        file: std::fs::File,
    }

    unsafe impl Send for Inner {}
    unsafe impl Sync for Inner {}

    extern "C" {
        fn flock(fd: i32, operation: i32) -> i32;
    }

    const LOCK_EX: i32 = 2;
    const LOCK_NB: i32 = 4;

    pub(crate) fn acquire(path: &Path) -> Result<Inner> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        let rc = unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
        if rc != 0 {
            return Err(Error::Locked(
                "database directory is locked by another process".into(),
            ));
        }
        Ok(Inner { file })
    }

    impl Drop for Inner {
        fn drop(&mut self) {
            let _ = flock(self.file.as_raw_fd(), 8); // LOCK_UN
        }
    }
}

use imp::Inner;

impl FileLock {
    pub fn acquire(path: &Path) -> Result<FileLock> {
        Ok(FileLock {
            inner: imp::acquire(path)?,
        })
    }
}

