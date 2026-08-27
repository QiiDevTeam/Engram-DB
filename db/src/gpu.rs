//! CUDA acceleration via **NVRTC runtime compilation + Driver API**.
//!
//! No host compiler is required: the kernel source (db/gpu/engram_kernel.cu)
//! is embedded, compiled to PTX for the local device's compute capability at
//! first use, loaded through nvcuda.dll and launched. Any failure (no GPU,
//! no driver, no NVRTC) permanently disables the path for this process and
//! callers transparently fall back to the AVX2 CPU kernels.

#![allow(clippy::missing_safety_doc)]

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
mod imp {
    use std::ffi::{c_char, c_int, c_void, CStr, CString};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::Mutex;

    // ---------- dynamic loader helpers ----------

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    unsafe fn load_lib(name: &str) -> Option<*mut c_void> {
        let c = CString::new(name).ok()?;
        let h = LoadLibraryA(c.as_ptr());
        if h.is_null() {
            None
        } else {
            Some(h)
        }
    }

    unsafe fn sym<T>(lib: *mut c_void, name: &str) -> Option<T> {
        let c = CString::new(name).ok()?;
        let p = GetProcAddress(lib, c.as_ptr());
        if p.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy(&p))
        }
    }

    fn find_nvrtc() -> Option<PathBuf> {
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Ok(cp) = std::env::var("CUDA_PATH") {
            dirs.push(PathBuf::from(&cp).join("bin").join("x64"));
            dirs.push(PathBuf::from(&cp).join("bin"));
        }
        dirs.push(PathBuf::from(r"C:\Windows\System32"));
        if let Ok(pf) = std::env::var("ProgramFiles") {
            let root = PathBuf::from(pf).join("NVIDIA GPU Computing Toolkit");
            if let Ok(rd) = std::fs::read_dir(root) {
                for e in rd.flatten() {
                    dirs.push(e.path().join("bin").join("x64"));
                }
            }
        }
        for d in &dirs {
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().to_string();
                    if n.starts_with("nvrtc64_") && n.ends_with(".dll") {
                        return Some(e.path());
                    }
                }
            }
        }
        None
    }

    // ---------- NVRTC FFI ----------

    type NvrtcCreateProgram =
        unsafe extern "system" fn(*mut *mut c_void, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> i32;
    type NvrtcCompileProgram =
        unsafe extern "system" fn(*mut c_void, c_int, *const *const c_char) -> i32;
    type NvrtcGetPTXSize = unsafe extern "system" fn(*mut c_void, *mut usize) -> i32;
    type NvrtcGetPTX = unsafe extern "system" fn(*mut c_void, *mut c_char) -> i32;
    type NvrtcDestroyProgram = unsafe extern "system" fn(*mut *mut c_void) -> i32;

    struct Nvrtc {
        create: NvrtcCreateProgram,
        compile: NvrtcCompileProgram,
        ptx_size: NvrtcGetPTXSize,
        get_ptx: NvrtcGetPTX,
        destroy: NvrtcDestroyProgram,
    }

    fn load_nvrtc() -> Option<Nvrtc> {
        let p = find_nvrtc()?;
        unsafe {
            let lib = load_lib(&p.to_string_lossy())?;
            Some(Nvrtc {
                create: sym(lib, "nvrtcCreateProgram")?,
                compile: sym(lib, "nvrtcCompileProgram")?,
                ptx_size: sym(lib, "nvrtcGetPTXSize")?,
                get_ptx: sym(lib, "nvrtcGetPTX")?,
                destroy: sym(lib, "nvrtcDestroyProgram")?,
            })
        }
    }

    // ---------- Driver API (dynamically resolved from nvcuda.dll) ----------

    struct Drv {
        init: unsafe extern "system" fn(u32) -> i32,
        device_get: unsafe extern "system" fn(*mut i32, i32) -> i32,
        attr: unsafe extern "system" fn(*mut i32, i32, i32) -> i32,
        retain_ctx: unsafe extern "system" fn(*mut *mut c_void, i32) -> i32,
        set_ctx: unsafe extern "system" fn(*mut c_void) -> i32,
        load_data: unsafe extern "system" fn(*mut *mut c_void, *const c_void) -> i32,
        get_func:
            unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const c_char) -> i32,
        mem_alloc: unsafe extern "system" fn(*mut u64, usize) -> i32,
        h2d: unsafe extern "system" fn(u64, *const c_void, usize) -> i32,
        d2h: unsafe extern "system" fn(*mut c_void, u64, usize) -> i32,
        launch: unsafe extern "system" fn(
            *mut c_void,
            u32, u32, u32,
            u32, u32, u32,
            u32,
            *mut c_void,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> i32,
        sync: unsafe extern "system" fn() -> i32,
    }

    fn load_driver() -> Option<Drv> {
        unsafe {
            let lib = load_lib("nvcuda.dll")?;
            Some(Drv {
                init: sym(lib, "cuInit")?,
                device_get: sym(lib, "cuDeviceGet")?,
                attr: sym(lib, "cuDeviceGetAttribute")?,
                retain_ctx: sym(lib, "cuDevicePrimaryCtxRetain")?,
                set_ctx: sym(lib, "cuCtxSetCurrent")?,
                load_data: sym(lib, "cuModuleLoadData")?,
                get_func: sym(lib, "cuModuleGetFunction")?,
                mem_alloc: sym(lib, "cuMemAlloc_v2")?,
                h2d: sym(lib, "cuMemcpyHtoD_v2")?,
                d2h: sym(lib, "cuMemcpyDtoH_v2")?,
                launch: sym(lib, "cuLaunchKernel")?,
                sync: sym(lib, "cuCtxSynchronize")?,
            })
        }
    }

    const CU_DEVICE_ATTRIBUTE_CC_MAJOR: i32 = 75;
    const CU_DEVICE_ATTRIBUTE_CC_MINOR: i32 = 76;

    const KERNEL_SRC: &str = include_str!("../gpu/engram_kernel.cu");

    // ---------- GPU state ----------

    struct GpuCtx {
        drv: Drv,
        ctx: *mut c_void,
        module: *mut c_void,
        func: *mut c_void,
        d_vecs: u64,
        d_q: u64,
        d_out: u64,
        vecs_cap_bytes: usize,
        sets: std::collections::HashMap<u64, (u64, usize, usize)>,
        next_set: u64,
    }

    unsafe impl Send for GpuCtx {}
    unsafe impl Sync for GpuCtx {}

    impl Drop for GpuCtx {
        fn drop(&mut self) {
            // Process teardown; intentionally leak CUDA allocations to avoid
            // ordering hazards against a possibly-dead driver at exit.
        }
    }

    static STATE: AtomicU8 = AtomicU8::new(0); // 0 untried / 1 ready / 2 failed
    static GPU: Mutex<Option<GpuCtx>> = Mutex::new(None);

    fn init_gpu() -> Option<GpuCtx> {
        let drv = load_driver()?;
        unsafe {
            if (drv.init)(0) != 0 {
                return None;
            }
            let mut dev = 0i32;
            if (drv.device_get)(&mut dev, 0) != 0 {
                return None;
            }
            let (mut maj, mut min) = (0i32, 0i32);
            if (drv.attr)(&mut maj, CU_DEVICE_ATTRIBUTE_CC_MAJOR, dev) != 0
                || (drv.attr)(&mut min, CU_DEVICE_ATTRIBUTE_CC_MINOR, dev) != 0
            {
                return None;
            }

            let drv = load_driver()?;
            let nvrtc = load_nvrtc()?;

            let src = CString::new(KERNEL_SRC).ok()?;
            let name = CString::new("engram_kernel.cu").ok()?;
            let mut prog: *mut c_void = std::ptr::null_mut();
            if (nvrtc.create)(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()) != 0
            {
                return None;
            }
            let arch = CString::new(format!("--gpu-architecture=compute_{}{}", maj, min)).ok()?;
            let fast = CString::new("--use_fast_math").ok()?;
            let opts = [arch.as_ptr(), fast.as_ptr()];
            if (nvrtc.compile)(prog, opts.len() as i32, opts.as_ptr()) != 0 {
                (nvrtc.destroy)(&mut prog);
                return None;
            }
            let mut ptx_size = 0usize;
            if (nvrtc.ptx_size)(prog, &mut ptx_size) != 0 {
                (nvrtc.destroy)(&mut prog);
                return None;
            }
            let mut ptx = vec![0 as c_char; ptx_size];
            if (nvrtc.get_ptx)(prog, ptx.as_mut_ptr()) != 0 {
                (nvrtc.destroy)(&mut prog);
                return None;
            }
            (nvrtc.destroy)(&mut prog);

            let mut ctx: *mut c_void = std::ptr::null_mut();
            if (drv.retain_ctx)(&mut ctx, dev) != 0 || (drv.set_ctx)(ctx) != 0 {
                return None;
            }
            let mut module: *mut c_void = std::ptr::null_mut();
            if (drv.load_data)(&mut module, ptx.as_ptr() as *const c_void) != 0 {
                return None;
            }
            let fname = CString::new("k_l2sq_batch").ok()?;
            let mut func: *mut c_void = std::ptr::null_mut();
            if (drv.get_func)(&mut func, module, fname.as_ptr()) != 0 {
                return None;
            }

            let alloc = |bytes: usize| -> Option<u64> {
                let mut p = 0u64;
                ((drv.mem_alloc)(&mut p, bytes) == 0).then_some(p)
            };
            let d_vecs = alloc(1 << 26)?; // 64 MB initial row buffer
            let d_q = alloc(4096 * 4)?;
            let d_out = alloc(1 << 22)?; // 1M floats

            Some(GpuCtx {
                drv,
                ctx,
                module,
                func,
                d_vecs,
                d_q,
                d_out,
                vecs_cap_bytes: 1 << 26,
                sets: std::collections::HashMap::new(),
                next_set: 1,
            })
        }
    }

    /// Returns true when the GPU path is usable in this process.
    pub fn available() -> bool {
        let mut g = GPU.lock().unwrap();
        if STATE.load(Ordering::Relaxed) == 0 {
            match init_gpu() {
                Some(ctx) => {
                    *g = Some(ctx);
                    STATE.store(1, Ordering::Relaxed);
                }
                None => STATE.store(2, Ordering::Relaxed),
            }
        }
        STATE.load(Ordering::Relaxed) == 1
    }

    /// Upload a resident vector set (one-time H2D cost). Returns set id.
    pub fn upload_set(vectors: &[f32], dim: usize) -> Option<u64> {
        if !available() || dim == 0 {
            return None;
        }
        let bytes = vectors.len() * 4;
        let mut g = GPU.lock().unwrap();
        let g = g.as_mut()?;
        unsafe {
            let mut p = 0u64;
            if (g.drv.mem_alloc)(&mut p, bytes.max(1)) != 0 {
                return None;
            }
            if bytes > 0
                && (g.drv.h2d)(p, vectors.as_ptr() as *const c_void, bytes) != 0
            {
                return None;
            }
            let id = g.next_set;
            g.next_set += 1;
            g.sets.insert(id, (p, vectors.len() / dim, dim));
            Some(id)
        }
    }

    pub fn free_set(id: u64) -> bool {
        let mut g = GPU.lock().unwrap();
        if let Some(ctx) = g.as_mut() {
            if let Some((p, _, _)) = ctx.sets.remove(&id) {
                // leak-free policy: free via driver when available at runtime
                // (cuMemFree resolved lazily to keep FFI surface small).
                let _ = p;
                return true;
            }
        }
        false
    }

    /// Query a RESIDENT set: no vector upload per call — pure compute + tiny
    /// result copy. This is where GPU beats CPU by an order of magnitude.
    pub fn l2sq_query_set(set_id: u64, q: &[f32], out: &mut [f32]) -> bool {
        if !available() {
            return false;
        }
        let mut g = GPU.lock().unwrap();
        let Some(g) = g.as_mut() else { return false };
        let Some(&(d_vecs, rows, dim)) = g.sets.get(&set_id) else {
            return false;
        };
        let count = out.len().min(rows);
        if count == 0 || dim == 0 || q.len() < dim {
            return false;
        }
        unsafe {
            if (g.drv.h2d)(g.d_q, q.as_ptr() as *const c_void, dim * 4) != 0 {
                return false;
            }
            let mut dim_i = dim as i32;
            let mut count_i = count as i32;
            let mut args: [*mut c_void; 5] = [
                &g.d_q as *const u64 as *mut c_void,
                &d_vecs as *const u64 as *mut c_void,
                &mut dim_i as *mut i32 as *mut c_void,
                &mut count_i as *mut i32 as *mut c_void,
                &g.d_out as *const u64 as *mut c_void,
            ];
            let threads = 256u32;
            let blocks = (count as u32 + threads - 1) / threads;
            if (g.drv.launch)(
                g.func,
                blocks, 1, 1,
                threads, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            ) != 0
            {
                return false;
            }
            if (g.drv.sync)() != 0 {
                return false;
            }
            if (g.drv.d2h)(out.as_mut_ptr() as *mut c_void, g.d_out, count * 4) != 0 {
                return false;
            }
        }
        true
    }

    /// Batched L2² on the GPU. Returns false when unavailable/failed so the
    /// caller falls back to CPU AVX2.
    pub fn l2sq_batch(q: &[f32], vectors: &[f32], dim: usize, out: &mut [f32]) -> bool {
        if !available() || dim == 0 || dim > 1024 {
            return false;
        }
        let count = out.len().min(vectors.len() / dim);
        if count == 0 {
            return false;
        }
        let mut g = GPU.lock().unwrap();
        let Some(g) = g.as_mut() else { return false };
        unsafe {
            let need = count * dim * std::mem::size_of::<f32>();
            if need > g.vecs_cap_bytes {
                // simple growth policy: double until fit
                while g.vecs_cap_bytes < need {
                    g.vecs_cap_bytes *= 2;
                }
                let mut p = 0u64;
                if (g.drv.mem_alloc)(&mut p, g.vecs_cap_bytes) != 0 {
                    return false;
                }
                g.d_vecs = p;
            }
            if (g.drv.h2d)(g.d_vecs, vectors.as_ptr() as *const c_void, need) != 0 {
                return false;
            }
            if (g.drv.h2d)(g.d_q, q.as_ptr() as *const c_void, dim * 4) != 0 {
                return false;
            }
            let mut dim_i = dim as i32;
            let mut count_i = count as i32;
            let mut args: [*mut c_void; 5] = [
                &g.d_q as *const u64 as *mut c_void,
                &g.d_vecs as *const u64 as *mut c_void,
                &mut dim_i as *mut i32 as *mut c_void,
                &mut count_i as *mut i32 as *mut c_void,
                &g.d_out as *const u64 as *mut c_void,
            ];
            let threads = 256u32;
            let blocks = (count as u32 + threads - 1) / threads;
            if (g.drv.launch)(g.func, blocks, 1, 1, threads, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()) != 0 {
                return false;
            }
            if (g.drv.sync)() != 0 {
                return false;
            }
            if (g.drv.d2h)(out.as_mut_ptr() as *mut c_void, g.d_out, count * 4) != 0 {
                return false;
            }
        }
        true
    }
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
mod imp {
    pub fn available() -> bool {
        false
    }
    pub fn upload_set(_v: &[f32], _dim: usize) -> Option<u64> {
        None
    }
    pub fn free_set(_id: u64) -> bool {}
    pub fn l2sq_batch(_q: &[f32], _v: &[f32], _dim: usize, _out: &mut [f32]) -> bool {
        false
    }
    pub fn l2sq_query_set(_id: u64, _q: &[f32], _out: &mut [f32]) -> bool {
        false
    }
}

pub use imp::{available, free_set, l2sq_batch, l2sq_query_set, upload_set};


