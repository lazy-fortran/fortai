#include "fortai_cuda_fattn_backend.h"

#include "fattn-mma-f16.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int qwen_head_size = 256;
constexpr int qwen_value_size = 256;
constexpr int qwen_query_tile = 8;
constexpr int qwen_gqa_tile = 8;
constexpr int qwen_fattn_columns = qwen_query_tile * qwen_gqa_tile;

__global__ void qwen_causal_mask(__half *mask, int batch, int active_rows,
    int start_position) {
    const int key = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int query = static_cast<int>(blockIdx.y);
    if (query >= batch || key >= active_rows) return;
    mask[static_cast<size_t>(query) * active_rows + key] =
        key <= start_position + query ? __float2half(0.0f) :
        __float2half(-65504.0f);
}

__global__ void qwen_attention_gate(const float *__restrict__ query,
    float *__restrict__ output, int heads, int batch, int query_stride,
    int output_stride, int query_head_stride) {
    const int value = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int head = static_cast<int>(blockIdx.y);
    const int token = static_cast<int>(blockIdx.z);
    if (value >= qwen_value_size || head >= heads || token >= batch) return;
    const size_t query_offset = static_cast<size_t>(token) * query_stride +
        static_cast<size_t>(head) * query_head_stride + qwen_head_size + value;
    const size_t output_offset = static_cast<size_t>(token) * output_stride +
        static_cast<size_t>(head) * qwen_value_size + value;
    const float gate = 1.0f / (1.0f + expf(-query[query_offset]));
    output[output_offset] *= gate;
}

using qwen_fattn_kernel = decltype(&flash_attn_ext_f16<qwen_head_size,
    qwen_value_size, qwen_query_tile, qwen_gqa_tile, false, false>);

static int qwen_fattn_blocks(int batch, int key_value_heads,
    int multiprocessor_count) {
    const int tiles = ((batch + qwen_query_tile - 1) / qwen_query_tile) *
        key_value_heads;
    /* Match launch_fattn's Ada+ stream-K schedule.  Even when there are fewer
     * logical query/GQA tiles than resident blocks, all SMs split the active
     * K/V range and the fixup kernel combines their partials. */
    return tiles > 0 ? 2 * multiprocessor_count : 0;
}

} // namespace

extern "C" size_t fortai_cuda_qwen38_fattn_mask_bytes(int batch,
    int active_rows) {
    if (batch <= 0 || active_rows <= 0) return 0;
    return static_cast<size_t>(batch) * active_rows * sizeof(__half);
}

extern "C" size_t fortai_cuda_qwen38_fattn_meta_bytes(int batch,
    int key_value_heads, int active_rows, int multiprocessor_count) {
    if (batch <= 0 || key_value_heads <= 0 || active_rows <= 0 ||
        multiprocessor_count <= 0) return 0;
    const int blocks = qwen_fattn_blocks(batch, key_value_heads,
        multiprocessor_count);
    const int tiles = ((batch + qwen_query_tile - 1) / qwen_query_tile) *
        key_value_heads;
    if (blocks <= 0 || tiles % blocks == 0) return 0;
    return static_cast<size_t>(blocks) * qwen_fattn_columns *
        (2 + qwen_value_size / 2) * sizeof(float2);
}

extern "C" int fortai_cuda_qwen38_fattn_f16(const float *query,
    const void *key_f16, const void *value_f16, float *output,
    void *mask_f16, void *meta, size_t meta_bytes, int heads,
    int key_value_heads, int start_position, int batch, int active_rows,
    int query_stride, int output_stride, int query_head_stride,
    int multiprocessor_count, void *stream_pointer) {
    if (!query || !key_f16 || !value_f16 || !output || !mask_f16 ||
        !stream_pointer || heads != 6 * key_value_heads || batch <= 0 ||
        active_rows <= 0 || start_position < 0 || query_stride <= 0 ||
        output_stride < heads * qwen_value_size ||
        query_head_stride < 2 * qwen_head_size || multiprocessor_count <= 0)
        return static_cast<int>(cudaErrorInvalidValue);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    auto *mask = static_cast<__half *>(mask_f16);
    qwen_causal_mask<<<dim3((active_rows + 255) / 256, batch, 1), 256, 0,
        stream>>>(mask, batch, active_rows, start_position);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);

    /* These are the D=DV=256, ncols=64 Ampere-family values returned by
     * ggml_cuda_fattn_mma_get_config_ampere.  They are also compile-time
     * constants inside the instantiated kernel; using a smaller host-side
     * shared-memory layout corrupts the first small-query launch. */
    constexpr int nbatch_fa = 64;
    constexpr int nthreads = 128;
    constexpr int nwarps = nthreads / 32;
    constexpr int nbatch_k2 = 128;
    constexpr int nbatch_v2 = 128;
    constexpr int nbatch_combine = 64;
    constexpr size_t shared_kv = static_cast<size_t>(nbatch_fa) *
        (nbatch_k2 + 4 + nbatch_v2 + 4) * sizeof(half2);
    constexpr size_t shared_q = static_cast<size_t>(qwen_fattn_columns) *
        (qwen_head_size / 2 + 4) * sizeof(half2);
    constexpr size_t shared_mask = static_cast<size_t>(qwen_query_tile) *
        (nbatch_fa / 2 + 4) * sizeof(half2);
    constexpr size_t shared_combine = static_cast<size_t>(nwarps) * 16 *
        (nbatch_combine + 4) * sizeof(half2);
    constexpr size_t shared_bytes = shared_combine > shared_q ?
        (shared_combine > shared_kv + shared_mask ? shared_combine :
            shared_kv + shared_mask) :
        (shared_q > shared_kv + shared_mask ? shared_q : shared_kv + shared_mask);

    auto kernel = flash_attn_ext_f16<qwen_head_size, qwen_value_size,
        qwen_query_tile, qwen_gqa_tile, false, false>;
    error = cudaFuncSetAttribute(reinterpret_cast<qwen_fattn_kernel>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (error != cudaSuccess) return static_cast<int>(error);

    const int blocks = qwen_fattn_blocks(batch, key_value_heads,
        multiprocessor_count);
    const size_t required_meta = fortai_cuda_qwen38_fattn_meta_bytes(batch,
        key_value_heads, active_rows, multiprocessor_count);
    if (blocks <= 0 || (required_meta > 0 && (!meta || meta_bytes < required_meta)))
        return static_cast<int>(cudaErrorInvalidValue);

    const uint3 query_fastdiv = init_fastdiv_values(batch);
    const uint32_t head_log2 = 16;
    kernel<<<blocks, dim3(32, nwarps, 1), shared_bytes, stream>>>(
        reinterpret_cast<const char *>(query),
        reinterpret_cast<const char *>(key_f16),
        reinterpret_cast<const char *>(value_f16),
        reinterpret_cast<const char *>(mask), nullptr, nullptr, output,
        static_cast<float2 *>(meta), 1.0f / sqrtf(float(qwen_head_size)),
        0.0f, 1.0f, 1.0f, head_log2, 0.0f,
        qwen_head_size, query_fastdiv, heads, 1,
        query_stride * static_cast<int>(sizeof(float)),
        query_head_stride * static_cast<int>(sizeof(float)), 0,
        qwen_head_size, active_rows, key_value_heads, 1,
        key_value_heads * qwen_head_size * static_cast<int>(sizeof(__half)),
        qwen_head_size * static_cast<int>(sizeof(__half)), 0,
        key_value_heads * qwen_value_size * static_cast<int>(sizeof(__half)),
        qwen_value_size * static_cast<int>(sizeof(__half)), 0,
        batch, 1, 1, active_rows * static_cast<int>(sizeof(__half)), 0, 0);
    error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);

    const int tiles = ((batch + qwen_query_tile - 1) / qwen_query_tile) *
        key_value_heads;
    if (tiles % blocks != 0) {
        flash_attn_stream_k_fixup<qwen_value_size, qwen_query_tile,
            qwen_gqa_tile><<<dim3(blocks, qwen_query_tile, qwen_gqa_tile),
            qwen_value_size, 0, stream>>>(output, static_cast<float2 *>(meta),
            batch, heads, 1, active_rows, key_value_heads, nbatch_fa);
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
    }

    qwen_attention_gate<<<dim3((qwen_value_size + 255) / 256, heads, batch),
        256, 0, stream>>>(query, output, heads, batch, query_stride,
        output_stride, query_head_stride);
    return static_cast<int>(cudaGetLastError());
}
