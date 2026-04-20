// ---------------------------------------------------------------------------
// codebook_init.cu — K-means VQ codebook initialisation on device
//
// Entry point: codebook_init_kmeans()
// Runs K Lloyd iterations directly on device to avoid PCIe round-trips.
// Forgy seeding (random rows from calibration data) is done on the host
// before calling this function.
// ---------------------------------------------------------------------------

#include "vq_cache.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace truellm {

// ---------------------------------------------------------------------------
// kmeans_assign_kernel: for each calibration vector find nearest centroid.
//   d_data    : [n_samples, head_dim] F16
//   d_book    : [codebook_size, head_dim] F16
//   d_assign  : [n_samples] int32  (output)
// ---------------------------------------------------------------------------
static __global__ void kmeans_assign_kernel(
    const __half* __restrict__ d_data,
    const __half* __restrict__ d_book,
    int32_t*      __restrict__ d_assign,
    int n_samples, int codebook_size, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;

    const __half* vec  = d_data + i * head_dim;
    float best = FLT_MAX;
    int   best_c = 0;

    for (int c = 0; c < codebook_size; ++c) {
        const __half* entry = d_book + c * head_dim;
        float dist = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            float diff = __half2float(vec[d]) - __half2float(entry[d]);
            dist += diff * diff;
        }
        if (dist < best) { best = dist; best_c = c; }
    }
    d_assign[i] = best_c;
}

// ---------------------------------------------------------------------------
// kmeans_update_kernel: accumulate sums for centroid update.
//   d_sums  : [codebook_size, head_dim] F32  (zeroed before call)
//   d_counts: [codebook_size] int32           (zeroed before call)
// ---------------------------------------------------------------------------
static __global__ void kmeans_update_kernel(
    const __half*  __restrict__ d_data,
    const int32_t* __restrict__ d_assign,
    float*         __restrict__ d_sums,
    int32_t*       __restrict__ d_counts,
    int n_samples, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;

    int c = d_assign[i];
    atomicAdd(&d_counts[c], 1);
    const __half* vec = d_data + i * head_dim;
    float*        sum = d_sums + c * head_dim;
    for (int d = 0; d < head_dim; ++d)
        atomicAdd(&sum[d], __half2float(vec[d]));
}

// ---------------------------------------------------------------------------
// kmeans_normalise_kernel: divide accumulated sums by counts → new centroids.
// ---------------------------------------------------------------------------
static __global__ void kmeans_normalise_kernel(
    __half*        __restrict__ d_book,
    const float*   __restrict__ d_sums,
    const int32_t* __restrict__ d_counts,
    int codebook_size, int head_dim)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= codebook_size) return;

    int cnt = d_counts[c];
    if (cnt == 0) return;   // empty cluster — keep old centroid
    float inv = 1.f / static_cast<float>(cnt);
    for (int d = 0; d < head_dim; ++d)
        d_book[c * head_dim + d] = __float2half(d_sums[c * head_dim + d] * inv);
}

// ---------------------------------------------------------------------------
// codebook_init_kmeans — host entry point
//
// d_data       : [n_samples, head_dim] F16  calibration vectors (device)
// d_book       : [codebook_size, head_dim] F16  (device) — pre-seeded, updated in-place
// n_iterations : Lloyd iterations to run
// ---------------------------------------------------------------------------
// Signature uses void* to match kv_compression_pass.h declaration — avoids
// pulling cuda_fp16.h into MSVC-compiled translation units.
void codebook_init_kmeans(
    const void*  d_data_v,
    void*        d_book_v,
    int n_samples, int codebook_size, int head_dim,
    int n_iterations, cudaStream_t stream)
{
    const __half* d_data = reinterpret_cast<const __half*>(d_data_v);
    __half*       d_book = reinterpret_cast<__half*>(d_book_v);

    // Temporary device buffers
    int32_t* d_assign  = nullptr;
    float*   d_sums    = nullptr;
    int32_t* d_counts  = nullptr;

    cudaMalloc(&d_assign, static_cast<size_t>(n_samples)               * sizeof(int32_t));
    cudaMalloc(&d_sums,   static_cast<size_t>(codebook_size * head_dim) * sizeof(float));
    cudaMalloc(&d_counts, static_cast<size_t>(codebook_size)            * sizeof(int32_t));

    int threads = 256;
    int blk_s   = (n_samples      + threads - 1) / threads;
    int blk_c   = (codebook_size  + threads - 1) / threads;

    for (int it = 0; it < n_iterations; ++it) {
        // E-step: assign
        kmeans_assign_kernel<<<blk_s, threads, 0, stream>>>(
            d_data, d_book, d_assign, n_samples, codebook_size, head_dim);

        // M-step: accumulate
        cudaMemsetAsync(d_sums,   0,
            static_cast<size_t>(codebook_size * head_dim) * sizeof(float), stream);
        cudaMemsetAsync(d_counts, 0,
            static_cast<size_t>(codebook_size) * sizeof(int32_t), stream);
        kmeans_update_kernel<<<blk_s, threads, 0, stream>>>(
            d_data, d_assign, d_sums, d_counts, n_samples, head_dim);

        // Normalise
        kmeans_normalise_kernel<<<blk_c, threads, 0, stream>>>(
            d_book, d_sums, d_counts, codebook_size, head_dim);
    }

    cudaStreamSynchronize(stream);
    cudaFree(d_assign);
    cudaFree(d_sums);
    cudaFree(d_counts);
}

} // namespace truellm
