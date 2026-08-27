// EngramDB device kernels — NVRTC-compilable (no host code, no includes).
//
// Runtime path: db/src/gpu.rs embeds this file, compiles it with NVRTC for
// the local device's compute capability, loads the PTX via the CUDA Driver
// API and launches k_l2sq_batch. No host compiler (MSVC) is involved.

extern "C" __global__ void k_l2sq_batch(const float *__restrict__ q,
                                        const float *__restrict__ vecs,
                                        int dim, int count,
                                        float *__restrict__ out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const float *v = vecs + (size_t)row * dim;
    float acc = 0.0f;
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        float d0 = q[i] - v[i];
        float d1 = q[i + 1] - v[i + 1];
        float d2 = q[i + 2] - v[i + 2];
        float d3 = q[i + 3] - v[i + 3];
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (; i < dim; ++i) {
        float d = q[i] - v[i];
        acc += d * d;
    }
    out[row] = acc;
}

extern "C" __global__ void k_dot_batch(const float *__restrict__ q,
                                       const float *__restrict__ vecs,
                                       int dim, int count,
                                       float *__restrict__ out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const float *v = vecs + (size_t)row * dim;
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) acc += q[i] * v[i];
    out[row] = acc;
}
