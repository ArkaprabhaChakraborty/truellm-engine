// ---------------------------------------------------------------------------
// kivi_cache.cu — KiVi outlier + INT4 group-quantized KV compression
// ---------------------------------------------------------------------------

#include "kivi_cache.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace truellm {

// ---------------------------------------------------------------------------
// kivi_compress_kernel
//
// One thread per KV vector (tok × head pair).
// Layout written to d_out per vector:
//   [n_outliers × {uint8 channel_idx, __half value}]   = n_outliers * 3 bytes
//   [n_groups   × {INT4 × group_size packed, __half scale, __half zero}]
//                                                       = n_groups * (group_size/2 + 4)
// Outlier selection: top-n_outliers by |x| via insertion sort.
// Residual quantization: per-group min-max linear INT4.
// ---------------------------------------------------------------------------
extern "C" __global__ void kivi_compress_kernel(
    const __half* __restrict__ d_kv_in,
    uint8_t*      __restrict__ d_out,
    int n_vecs, int head_dim, int n_outliers, int group_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_vecs) return;

    // Defensive clamp: host callers should never exceed these, but if they do,
    // silently truncate rather than smash the stack arrays sized to the caps.
    if (head_dim <= 0 || head_dim > KIVI_MAX_HEAD_DIM) return;
    if (group_size <= 0) return;

    const __half* src = d_kv_in + tid * head_dim;
    size_t out_stride = kivi_compressed_bytes(head_dim, n_outliers, group_size);
    uint8_t* dst = d_out + tid * out_stride;

    // ---- Pass 1: find top-n_outliers by |value| via insertion sort --------
    // outlier_idx[i] holds channel indices of the n_outliers largest |values|
    uint8_t outlier_idx[KIVI_MAX_OUTLIERS];
    float   outlier_mag[KIVI_MAX_OUTLIERS];
    int n_out = (n_outliers < KIVI_MAX_OUTLIERS) ? n_outliers : KIVI_MAX_OUTLIERS;
    if (n_out < 0) n_out = 0;
    if (n_out > head_dim) n_out = head_dim;
    for (int i = 0; i < n_out; ++i) { outlier_idx[i] = 0; outlier_mag[i] = -1.f; }

    float min_mag = -1.f;
    int   min_pos = 0;

    for (int d = 0; d < head_dim; ++d) {
        float v = fabsf(__half2float(src[d]));
        if (v > min_mag) {
            outlier_idx[min_pos] = static_cast<uint8_t>(d);
            outlier_mag[min_pos] = v;
            // Update min
            min_mag = outlier_mag[0]; min_pos = 0;
            for (int i = 1; i < n_out; ++i) {
                if (outlier_mag[i] < min_mag) { min_mag = outlier_mag[i]; min_pos = i; }
            }
        }
    }

    // Build a boolean mask for outlier channels. head_dim is already bounded
    // above by KIVI_MAX_HEAD_DIM, so every outlier_idx entry is a valid index.
    bool is_outlier[KIVI_MAX_HEAD_DIM] = {};
    for (int i = 0; i < n_out; ++i) {
        int ch = outlier_idx[i];
        if (ch >= 0 && ch < head_dim) is_outlier[ch] = true;
    }

    // ---- Write outlier section --------------------------------------------
    uint8_t* p = dst;
    for (int i = 0; i < n_out; ++i) {
        *p++ = outlier_idx[i];
        __half val = src[outlier_idx[i]];
        memcpy(p, &val, 2); p += 2;
    }

    // ---- Pass 2: INT4 group-quantize residual channels -------------------
    int n_res    = head_dim - n_out;
    int n_groups = (n_res + group_size - 1) / group_size;

    // Collect residual channel values
    float residuals[KIVI_MAX_HEAD_DIM];
    int   r_idx = 0;
    for (int d = 0; d < head_dim && r_idx < n_res; ++d) {
        if (!is_outlier[d]) residuals[r_idx++] = __half2float(src[d]);
    }

    for (int g = 0; g < n_groups; ++g) {
        int g_start = g * group_size;
        int g_end   = g_start + group_size;
        if (g_end > n_res) g_end = n_res;

        float gmin =  FLT_MAX, gmax = -FLT_MAX;
        for (int i = g_start; i < g_end; ++i) {
            if (residuals[i] < gmin) gmin = residuals[i];
            if (residuals[i] > gmax) gmax = residuals[i];
        }
        float range = gmax - gmin;
        bool  degenerate = !(range > 1e-6f);  // true range, NaN, or Inf
        float scale = degenerate ? 0.f : (15.f / range);

        // Pack INT4 pairs. Degenerate groups quantize to 0 — decompress with
        // scale=0 reconstructs `gmin` exactly (which equals gmax in this case).
        for (int i = g_start; i < g_end; i += 2) {
            int q0 = (i     < g_end) ? __float2int_rn((residuals[i]   - gmin) * scale) : 0;
            int q1 = (i + 1 < g_end) ? __float2int_rn((residuals[i+1] - gmin) * scale) : 0;
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            *p++ = static_cast<uint8_t>((q1 << 4) | q0);
        }
        // Write scale (post-reconstruction step size, 0 when degenerate) + zero.
        float step = degenerate ? 0.f : (1.f / scale);
        // Sanitise gmin: reject NaN (gmin != gmin) and Inf.  Decompress
        // would otherwise produce NaN/Inf across the entire group.
        bool gmin_ok = (gmin == gmin) && (gmin > -FLT_MAX) && (gmin < FLT_MAX);
        __half hscale = __float2half(step);
        __half hzero  = __float2half(gmin_ok ? gmin : 0.f);
        memcpy(p, &hscale, 2); p += 2;
        memcpy(p, &hzero,  2); p += 2;
    }
}

// ---------------------------------------------------------------------------
// kivi_decompress_kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void kivi_decompress_kernel(
    const uint8_t* __restrict__ d_in,
    __half*        __restrict__ d_kv_out,
    int n_vecs, int head_dim, int n_outliers, int group_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_vecs) return;

    if (head_dim <= 0 || head_dim > KIVI_MAX_HEAD_DIM) return;
    if (group_size <= 0) return;

    int n_out = (n_outliers < KIVI_MAX_OUTLIERS) ? n_outliers : KIVI_MAX_OUTLIERS;
    if (n_out < 0) n_out = 0;
    if (n_out > head_dim) n_out = head_dim;

    size_t in_stride = kivi_compressed_bytes(head_dim, n_outliers, group_size);
    const uint8_t* src = d_in    + tid * in_stride;
    __half*        dst = d_kv_out + tid * head_dim;

    float result[KIVI_MAX_HEAD_DIM] = {};
    bool  written[KIVI_MAX_HEAD_DIM] = {};

    // ---- Read outlier section --------------------------------------------
    const uint8_t* p = src;
    for (int i = 0; i < n_out; ++i) {
        uint8_t ch = *p++;
        __half  val; memcpy(&val, p, 2); p += 2;
        if (ch < head_dim) { result[ch] = __half2float(val); written[ch] = true; }
    }

    // ---- Dequantize residuals -------------------------------------------
    int n_res    = head_dim - n_out;
    int n_groups = (n_res + group_size - 1) / group_size;

    // Collect residual channel indices
    int res_channels[KIVI_MAX_HEAD_DIM];
    int r_idx = 0;
    for (int d = 0; d < head_dim && r_idx < n_res; ++d)
        if (!written[d]) res_channels[r_idx++] = d;

    for (int g = 0; g < n_groups; ++g) {
        int g_start = g * group_size;
        int g_end   = g_start + group_size;
        if (g_end > n_res) g_end = n_res;

        // Read packed INT4 pairs first, then scale+zero at end
        const uint8_t* pack = p;
        int pack_bytes = (g_end - g_start + 1) / 2;
        p += pack_bytes;
        __half hscale, hzero;
        memcpy(&hscale, p, 2); p += 2;
        memcpy(&hzero,  p, 2); p += 2;
        float scale = __half2float(hscale);
        float zero  = __half2float(hzero);

        for (int i = g_start; i < g_end; ++i) {
            int byte_off = (i - g_start) / 2;
            int nibble   = ((i - g_start) % 2 == 0)
                         ? (pack[byte_off] & 0x0F)
                         : (pack[byte_off] >> 4);
            float v = static_cast<float>(nibble) * scale + zero;
            int ch = res_channels[i];
            if (ch >= 0 && ch < head_dim) result[ch] = v;
        }
    }

    for (int d = 0; d < head_dim; ++d)
        dst[d] = __float2half(result[d]);
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
// Returns true iff the given (head_dim, n_outliers, group_size) triple
// is compatible with the on-stack buffers inside the KIVI kernels.
bool kivi_params_supported(int head_dim, int n_outliers, int group_size)
{
    return head_dim   > 0 && head_dim   <= KIVI_MAX_HEAD_DIM
        && n_outliers >= 0 && n_outliers <= KIVI_MAX_OUTLIERS
        && group_size > 0
        && n_outliers <= head_dim;
}

void kivi_compress(
    const __half* d_kv_in,
    uint8_t*      d_out,
    int n_vecs, int head_dim, int n_outliers, int group_size,
    cudaStream_t stream)
{
    if (n_vecs <= 0) return;
    if (!kivi_params_supported(head_dim, n_outliers, group_size)) return;
    int threads = 128;
    int blocks  = (n_vecs + threads - 1) / threads;
    kivi_compress_kernel<<<blocks, threads, 0, stream>>>(
        d_kv_in, d_out, n_vecs, head_dim, n_outliers, group_size);
}

void kivi_decompress(
    const uint8_t* d_in,
    __half*        d_kv_out,
    int n_vecs, int head_dim, int n_outliers, int group_size,
    cudaStream_t stream)
{
    if (n_vecs <= 0) return;
    if (!kivi_params_supported(head_dim, n_outliers, group_size)) return;
    int threads = 128;
    int blocks  = (n_vecs + threads - 1) / threads;
    kivi_decompress_kernel<<<blocks, threads, 0, stream>>>(
        d_in, d_kv_out, n_vecs, head_dim, n_outliers, group_size);
}

} // namespace truellm
