//! GPU (NVRTC + Driver API) availability, correctness & resident-set probe.

use engram_db::gpu;
use engram_db::simd;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0 >> 11
    }
}

const DIM: usize = 128;

fn percentile(v: &mut [u128], p: f64) -> u128 {
    v.sort_unstable();
    let idx = ((v.len() as f64 - 1.0) * p).round() as usize;
    v[idx.min(v.len() - 1)]
}

fn main() {
    let avail = gpu::available();
    println!("gpu path available: {avail}");
    if !avail {
        println!("falling back to CPU AVX2 (no GPU/NVRTC found)");
        return;
    }

    let count = 100_000usize;
    let mut r = Rng(42);
    let q: Vec<f32> = (0..DIM)
        .map(|_| ((r.next() % 10000) as f32 / 10000.0 - 0.5))
        .collect();
    let vecs: Vec<f32> = (0..count * DIM)
        .map(|_| ((r.next() % 10000) as f32 / 10000.0 - 0.5))
        .collect();

    // ---- one-shot batch (includes H2D upload) ----
    let mut gpu_out = vec![f32::NAN; count];
    let t = std::time::Instant::now();
    assert!(gpu::l2sq_batch(&q, &vecs, DIM, &mut gpu_out));
    let cold_us = t.elapsed().as_micros();

    let t = std::time::Instant::now();
    let mut cpu_out = vec![0f32; count];
    simd::l2_sq_batch_contig(&q, &vecs, DIM, &mut cpu_out);
    let cpu_us = t.elapsed().as_micros();

    let mut max_err = 0f32;
    for i in 0..count {
        let e = ((gpu_out[i] - cpu_out[i]) / cpu_out[i].max(1e-9)).abs();
        max_err = max_err.max(e);
    }
    println!(
        "one-shot (incl upload): gpu {cold_us} us vs cpu {cpu_us} us | max rel err {max_err:.2e}"
    );

    // ---- RESIDENT set: upload once, then query many ----
    let set = gpu::upload_set(&vecs, DIM).expect("resident upload");
    let mut lat: Vec<u128> = Vec::new();
    let mut qr = Rng(9);
    for i in 0..50 {
        let qq: Vec<f32> = (0..DIM)
            .map(|_| ((qr.next() % 10000) as f32 / 10000.0 - 0.5))
            .collect();
        let t = std::time::Instant::now();
        assert!(gpu::l2sq_query_set(set, &qq, &mut gpu_out));
        lat.push(t.elapsed().as_micros());

        if i == 0 {
            // numeric cross-check on first resident query
            let mut ref0 = vec![0f32; count];
            simd::l2_sq_batch_contig(&qq, &vecs, DIM, &mut ref0);
            let e = ((gpu_out[12345] - ref0[12345]) / ref0[12345].max(1e-9)).abs();
            assert!(e < 1e-4, "resident numeric error {e}");
        }
        let _ = i;
    }
    println!(
        "resident query: p50={} us  p99={} us   (cpu same workload ≈ {cpu_us} us)",
        percentile(lat.clone().as_mut(), 0.5),
        percentile(lat.as_mut(), 0.99),
    );
    gpu::free_set(set);
}
