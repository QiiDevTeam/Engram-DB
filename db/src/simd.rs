//! Runtime-dispatched SIMD kernels: ternary-sketch popcount dot + f32 L2/dot.
//! AVX2 paths via std::arch, scalar fallbacks everywhere else.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn cpu_has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Ternary sketch dot over packed 2-bit lanes, bit-plane formulation:
/// acc = |a+∩b+| + |a-∩b-| - |a+∩b-| - |a-∩b+|
/// where low bit-plane = +1 flags, high bit-plane = -1 flags.
/// `a_words`/`b_words` must have equal length; returns raw match count.
pub fn ternary_dot_count(a_words: &[u64], b_words: &[u64]) -> i64 {
    debug_assert_eq!(a_words.len(), b_words.len());
    let n = a_words.len();

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && n >= 4 {
            return unsafe { ternary_dot_count_avx2(a_words, b_words) };
        }
    }
    ternary_dot_count_scalar(a_words, b_words)
}

fn ternary_dot_count_scalar(a_words: &[u64], b_words: &[u64]) -> i64 {
    let mut acc: i64 = 0;
    for (&a, &b) in a_words.iter().zip(b_words.iter()) {
        let a_lo = a & 0x5555_5555_5555_5555;
        let b_lo = b & 0x5555_5555_5555_5555;
        let a_hi = (a >> 1) & 0x5555_5555_5555_5555;
        let b_hi = (b >> 1) & 0x5555_5555_5555_5555;
        acc += ((a_lo & b_lo).count_ones() as i64)
            + ((a_hi & b_hi).count_ones() as i64)
            - ((a_lo & b_hi).count_ones() as i64)
            - ((a_hi & b_lo).count_ones() as i64);
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ternary_dot_count_avx2(a_words: &[u64], b_words: &[u64]) -> i64 {
    let lo = _mm256_set1_epi64x(0x5555_5555_5555_5555);
    let mut acc: i64 = 0;
    let mut i = 0;
    let n = a_words.len();
    while i + 4 <= n {
        let a = _mm256_loadu_si256(a_words.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(b_words.as_ptr().add(i) as *const __m256i);
        let a_lo = _mm256_and_si256(a, lo);
        let b_lo = _mm256_and_si256(b, lo);
        let a_hi = _mm256_and_si256(_mm256_srli_epi64(a, 1), lo);
        let b_hi = _mm256_and_si256(_mm256_srli_epi64(b, 1), lo);
        acc += popcnt4(_mm256_and_si256(a_lo, b_lo));
        acc -= popcnt4(_mm256_and_si256(a_lo, b_hi));
        acc += popcnt4(_mm256_and_si256(a_hi, b_hi));
        acc -= popcnt4(_mm256_and_si256(a_hi, b_lo));
        i += 4;
    }
    if i < n {
        acc += ternary_dot_count_scalar(&a_words[i..], &b_words[i..]);
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn popcnt4(v: __m256i) -> i64 {
    let mut out = [0u64; 4];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
    (out[0].count_ones() + out[1].count_ones() + out[2].count_ones() + out[3].count_ones()) as i64
}

/// Sum of squared differences between two equal-length float slices.
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && a.len() >= 8 {
            return unsafe { l2_sq_avx2(a, b) };
        }
    }
    l2_sq_scalar(a, b)
}

fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    let n = a.len();
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let d = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += 8;
    }
    let mut sum = horizontal_sum(acc);
    sum += l2_sq_scalar(&a[i..], &b[i..]);
    sum
}

/// Dot product of two equal-length float slices.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && a.len() >= 8 {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    dot_f32_scalar(a, b)
}

fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    let n = a.len();
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    horizontal_sum(acc) + dot_f32_scalar(&a[i..], &b[i..])
}

/// Batched L2² of one query against `rows` contiguous rows (row stride = dim),
/// mirroring Go simd.BatchL2Sq4Contig: processes 4 rows per iteration for ILP.
pub fn l2_sq_batch_contig(query: &[f32], vectors: &[f32], dim: usize, out: &mut [f32]) {
    let count = out.len().min(vectors.len() / dim.max(1));
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && dim >= 8 && count >= 4 {
            unsafe { l2_sq_batch4_avx2(query, vectors, dim, count, out) };
            return;
        }
    }
    for r in 0..count {
        out[r] = l2_sq(query, &vectors[r * dim..(r + 1) * dim]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_sq_batch4_avx2(query: &[f32], vectors: &[f32], dim: usize, count: usize, out: &mut [f32]) {
    let mut r = 0;
    while r + 4 <= count {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= dim {
            let vq = _mm256_loadu_ps(query.as_ptr().add(i));
            acc0 = fmadd_sub(acc0, vq, vectors, dim, r, i, 0);
            acc1 = fmadd_sub(acc1, vq, vectors, dim, r, i, 1);
            acc2 = fmadd_sub(acc2, vq, vectors, dim, r, i, 2);
            acc3 = fmadd_sub(acc3, vq, vectors, dim, r, i, 3);
            i += 8;
        }
        let base = r * dim;
        let tail = l2_sq_scalar(&query[i.min(dim)..], &vectors[base + i.min(dim)..base + dim]);
        out[r] = horizontal_sum(acc0) + tail;
        for (k, acc) in [acc1, acc2, acc3].into_iter().enumerate() {
            let rb = (r + 1 + k) * dim;
            let t = l2_sq_scalar(&query[i.min(dim)..], &vectors[rb + i.min(dim)..rb + dim]);
            out[r + 1 + k] = horizontal_sum(acc) + t;
        }
        r += 4;
    }
    while r < count {
        out[r] = l2_sq(query, &vectors[r * dim..(r + 1) * dim]);
        r += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fmadd_sub(
    acc: __m256,
    vq: __m256,
    vectors: &[f32],
    dim: usize,
    row: usize,
    off: usize,
    delta: usize,
) -> __m256 {
    let base = (row + delta) * dim + off;
    let vr = _mm256_loadu_ps(vectors.as_ptr().add(base));
    let d = _mm256_sub_ps(vq, vr);
    _mm256_fmadd_ps(d, d, acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(hi, lo);
    let sh = _mm_movehl_ps(s, s);
    let s2 = _mm_add_ps(s, sh);
    let sh2 = _mm_shuffle_ps(s2, s2, 0x55);
    let s3 = _mm_add_ss(s2, sh2);
    _mm_cvtss_f32(s3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_matches_reference() {
        // lanes: a = [+,-,+,...], b = [-,-,+,..]
        let pack = |vals: &[i8]| -> Vec<u64> {
            vals.iter()
                .enumerate()
                .fold(vec![0u64; (vals.len() + 31) / 32], |mut w, (i, &v)| {
                    let c: u64 = match v {
                        1 => 1,
                        -1 => 2,
                        _ => 0,
                    };
                    w[i / 32] |= c << (2 * (i % 32));
                    w
                })
        };
        let a_vals: Vec<i8> = (0..256i64).map(|i| (i % 7).signum() as i8).collect();
        let b_vals: Vec<i8> = (0..256i64).map(|i| ((i % 5) - 2).signum() as i8).collect();
        let expect: i64 = a_vals
            .iter()
            .zip(b_vals.iter())
            .map(|(x, y)| (*x as i64) * (*y as i64))
            .sum();
        assert_eq!(ternary_dot_count(&pack(&a_vals), &pack(&b_vals)), expect);
    }

    #[test]
    fn l2_and_dot_correct() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i as f32) * 0.5).collect();
        let ref_l2: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let ref_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((l2_sq(&a, &b) - ref_l2).abs() < 1e-3);
        assert!((dot_f32(&a, &b) - ref_dot).abs() < 1e-2);
    }

    #[test]
    fn batch_matches_single() {
        let dim = 32;
        let q: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let count = 9;
        let vecs: Vec<f32> = (0..count * dim).map(|i| (i % 13) as f32 * 0.07).collect();
        let mut out = vec![0f32; count];
        l2_sq_batch_contig(&q, &vecs, dim, &mut out);
        for r in 0..count {
            let expect = l2_sq(&q, &vecs[r * dim..(r + 1) * dim]);
            assert!((out[r] - expect).abs() < 1e-3, "row {r}");
        }
    }
}

