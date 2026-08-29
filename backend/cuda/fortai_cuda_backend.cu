#include "fortai_cuda_backend.h"
#include "fortai_cuda_fattn_backend.h"

#include <cuda_fp16.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_topk.cuh>
#include <cuda/iterator>
#include <cuda/stream_ref>
#include <mma.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <new>
#include <vector>

using namespace nvcuda;

namespace {

constexpr int q8_block_width = 32;
constexpr int q8_block_bytes = 34;
constexpr int q4_block_width = 32;
constexpr int q4_block_bytes = 18;
constexpr int q8_activation_block_bytes = 36;
constexpr int q8_activation_data_offset = 4;
constexpr int mmq_grid_blocks = 36;
constexpr int mmq_rows_per_tile = 128;
constexpr int mmq_tokens_per_tile = 64;
constexpr int mmq_tile_elements = mmq_rows_per_tile * mmq_tokens_per_tile;
constexpr int qwen_graph_slots = 66;

struct fortai_cuda_q8_context_impl {
    int device = 0;
    int multiprocessor_count = 0;
    bool mma_available = false;
    cudaStream_t stream = nullptr;
    bool owns_stream = true;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    int8_t *scratch_activation = nullptr;
    size_t scratch_activation_bytes = 0;
    int8_t *scratch_raw = nullptr;
    size_t scratch_raw_bytes = 0;
    float *scratch_output = nullptr;
    size_t scratch_output_bytes = 0;
    float *scratch_aux = nullptr;
    size_t scratch_aux_bytes = 0;
    int8_t *scratch_batched_activation = nullptr;
    size_t scratch_batched_activation_bytes = 0;
    float *scratch_mmq_fixup = nullptr;
    size_t scratch_mmq_fixup_bytes = 0;
    int32_t *scratch_mmq_tiles = nullptr;
    size_t scratch_mmq_tiles_bytes = 0;
    int32_t *scratch_tokens = nullptr;
    size_t scratch_tokens_bytes = 0;
    /* llama.cpp's flash-attention path dequantizes only the active K/V view
     * into a temporary F16 pool before the tensor-core kernel.  Keep one
     * reusable pair per CUDA context rather than repeating Q8 decode for
     * every query tile (or retaining a second full-context cache). */
    __half *scratch_attention_key_f16 = nullptr;
    size_t scratch_attention_key_f16_bytes = 0;
    __half *scratch_attention_value_f16 = nullptr;
    size_t scratch_attention_value_f16_bytes = 0;
    __half *scratch_attention_mask_f16 = nullptr;
    size_t scratch_attention_mask_f16_bytes = 0;
    void *scratch_attention_meta = nullptr;
    size_t scratch_attention_meta_bytes = 0;
    /* Fortran allocatables are pageable.  Reusing one pinned staging buffer
     * avoids CUDA's per-call pageable registration in the logits/activation
     * download path while keeping the public ABI unchanged. */
    void *download_host = nullptr;
    size_t download_host_bytes = 0;
    size_t topk_temp_bytes = 0;
    int topk_temp_elements = 0;
    int topk_temp_k = 0;
    cudaStream_t topk_streams[32] = {};
    cudaEvent_t topk_ready = nullptr;
    cudaEvent_t topk_done[32] = {};
    int *position = nullptr;
    /* Host mirror used only for kernels whose ABI takes a scalar start
     * position.  The device pointer above remains the source of truth for
     * the pointer-taking attention kernels. */
    int position_value = 0;
    cudaGraph_t graphs[qwen_graph_slots] = {};
    cudaGraphExec_t graph_execs[qwen_graph_slots] = {};
    char error[256] = {};
};

struct fortai_cuda_q8_weights_impl {
    fortai_cuda_q8_context_impl *context = nullptr;
    int rows = 0;
    int width = 0;
    int blocks = 0;
    int8_t *device_data = nullptr;
};

static void launch_q8_repack_activation(const int8_t *raw, int8_t *padded,
    int blocks, cudaStream_t stream);

static int fail(fortai_cuda_q8_context_impl *context, int code,
    const char *operation, cudaError_t error = cudaSuccess) {
    if (context) {
        if (error == cudaSuccess)
            std::snprintf(context->error, sizeof(context->error), "%s", operation);
        else
            std::snprintf(context->error, sizeof(context->error), "%s: %s",
                operation, cudaGetErrorString(error));
    }
    return code;
}

__device__ __forceinline__ int load_i32(const int8_t *address) {
    const uint16_t *values = reinterpret_cast<const uint16_t *>(address);
    return static_cast<int>(values[0]) | (static_cast<int>(values[1]) << 16);
}

__device__ __forceinline__ float block_scale(const int8_t *block) {
    const uint16_t bits = *reinterpret_cast<const uint16_t *>(block);
    return __half2float(*reinterpret_cast<const __half *>(&bits));
}

/* Scalar decode kernel with the same block/lane ownership as llama.cpp's
 * MMVQ Q8_0×Q8_1 path.  The older GEMV kernels let one lane accumulate a whole
 * 32-value block and multiply its scale after the integer dot.  MMVQ splits a
 * block across four lanes and scales each partial dot before the warp
 * reduction; that reduction order is observable in long mixed-quantized
 * recurrent traces.  Keep the layout explicit here: Q8_0 has qi=8 packed
 * int32 words, vdr=2 words per lane, and four warps per output row. */
__global__ void q8_gemv_mmvq(const int8_t *__restrict__ weights,
    const int8_t *__restrict__ activation, int rows, int blocks,
    float *__restrict__ output) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (row >= rows) return;

    const int8_t *row_weights = weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    float accumulator = 0.0f;

    /* qi/vdr = 8/2 = 4 lanes own one Q8_0 block.  Four warps therefore
     * cover 32 blocks per iteration, exactly as MMVQ's generic ncols=1
     * launch on CUDA devices. */
    const int first_block = tid / 4;
    const int kqs = 2 * (tid & 3);
    for (int block = first_block; block < blocks; block += 32) {
        const int8_t *weight_block = row_weights + block * q8_block_bytes;
        const int8_t *activation_block = activation + block * q8_activation_block_bytes;
        int dot = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            dot = __dp4a(load_i32(weight_block + 2 + 4 * (kqs + i)),
                load_i32(activation_block + q8_activation_data_offset + 4 * (kqs + i)), dot);
        }
        /* Match vec_dot_q8_0_q8_1: scale each lane's integer partial before
         * the warp reduction instead of scaling a complete block afterwards. */
        accumulator += block_scale(weight_block) * block_scale(activation_block) *
            static_cast<float>(dot);
    }

    /* MMVQ stores each warp's *lane* partial before reduction.  Warp 0 then
     * adds the matching lanes from the other warps and performs one XOR
     * reduction over all four lanes' block contributions. */
    __shared__ float partial[4][32];
    partial[warp][lane] = accumulator;
    __syncthreads();
    if (warp == 0) {
        accumulator += partial[1][lane];
        accumulator += partial[2][lane];
        accumulator += partial[3][lane];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            accumulator += __shfl_xor_sync(0xffffffffu, accumulator, offset);
        if (lane == 0) output[row] = accumulator;
    }
}

static int validate_weights(const fortai_cuda_q8_weights_impl *weights,
    const void *activation, size_t activation_bytes, const void *output,
    size_t output_bytes) {
    if (!weights || !weights->device_data || !activation || !output) return FORTAI_CUDA_INVALID;
    const size_t expected_activation = static_cast<size_t>(weights->blocks) * q8_block_bytes;
    const size_t expected_output = static_cast<size_t>(weights->rows) * sizeof(float);
    if (activation_bytes < expected_activation || output_bytes < expected_output)
        return FORTAI_CUDA_INVALID;
    return FORTAI_CUDA_OK;
}

static void launch_q8(fortai_cuda_q8_weights_impl *weights,
    const int8_t *activation, float *output, cudaStream_t stream) {
    q8_gemv_mmvq<<<weights->rows, 128, 0, stream>>>(weights->device_data,
        activation, weights->rows, weights->blocks, output);
}

static cudaError_t ensure_host_matvec_scratch(fortai_cuda_q8_context_impl *context,
    size_t activation_bytes, size_t output_bytes) {
    const size_t blocks = activation_bytes / q8_block_bytes;
    const size_t padded_activation_bytes = blocks * q8_activation_block_bytes;
    if (activation_bytes > context->scratch_raw_bytes) {
        cudaError_t error = cudaFree(context->scratch_raw);
        if (error != cudaSuccess) return error;
        context->scratch_raw = nullptr;
        error = cudaMalloc(reinterpret_cast<void **>(&context->scratch_raw), activation_bytes);
        if (error != cudaSuccess) return error;
        context->scratch_raw_bytes = activation_bytes;
    }
    if (padded_activation_bytes > context->scratch_activation_bytes) {
        cudaError_t error = cudaFree(context->scratch_activation);
        if (error != cudaSuccess) return error;
        context->scratch_activation = nullptr;
        error = cudaMalloc(reinterpret_cast<void **>(&context->scratch_activation),
            padded_activation_bytes);
        if (error != cudaSuccess) return error;
        context->scratch_activation_bytes = padded_activation_bytes;
    }
    if (output_bytes > context->scratch_output_bytes) {
        cudaError_t error = cudaFree(context->scratch_output);
        if (error != cudaSuccess) return error;
        context->scratch_output = nullptr;
        error = cudaMalloc(reinterpret_cast<void **>(&context->scratch_output),
            output_bytes);
        if (error != cudaSuccess) return error;
        context->scratch_output_bytes = output_bytes;
    }
    return cudaSuccess;
}

/* Device-F32 matvecs quantize directly from a device activation.  Keep this
 * small reservation separate from the host matvec scratch: unlike the host
 * path it never needs the raw staging or output buffers, and the reservation
 * can be completed before CUDA graph capture starts. */
static cudaError_t ensure_device_matvec_scratch(fortai_cuda_q8_context_impl *context,
    size_t activation_elements) {
    const size_t blocks = (activation_elements + q8_block_width - 1) / q8_block_width;
    const size_t bytes = blocks * q8_activation_block_bytes;
    if (bytes <= context->scratch_activation_bytes && context->scratch_activation != nullptr)
        return cudaSuccess;
    cudaError_t error = cudaFree(context->scratch_activation);
    if (error != cudaSuccess) return error;
    context->scratch_activation = nullptr;
    context->scratch_activation_bytes = 0;
    error = cudaMalloc(reinterpret_cast<void **>(&context->scratch_activation), bytes);
    if (error == cudaSuccess) context->scratch_activation_bytes = bytes;
    return error;
}

static cudaError_t ensure_aux_scratch(fortai_cuda_q8_context_impl *context, size_t bytes) {
    if (bytes <= context->scratch_aux_bytes) return cudaSuccess;
    cudaError_t error = cudaFree(context->scratch_aux);
    if (error != cudaSuccess) return error;
    context->scratch_aux = nullptr;
    error = cudaMalloc(reinterpret_cast<void **>(&context->scratch_aux), bytes);
    if (error == cudaSuccess) context->scratch_aux_bytes = bytes;
    return error;
}

static cudaError_t ensure_batched_activation_scratch(fortai_cuda_q8_context_impl *context,
    size_t bytes) {
    if (bytes <= context->scratch_batched_activation_bytes &&
        context->scratch_batched_activation != nullptr) return cudaSuccess;
    cudaError_t error = cudaFree(context->scratch_batched_activation);
    if (error != cudaSuccess) return error;
    context->scratch_batched_activation = nullptr;
    context->scratch_batched_activation_bytes = 0;
    error = cudaMalloc(reinterpret_cast<void **>(&context->scratch_batched_activation), bytes);
    if (error == cudaSuccess) context->scratch_batched_activation_bytes = bytes;
    return error;
}

static cudaError_t ensure_mmq_fixup_scratch(fortai_cuda_q8_context_impl *context,
    size_t partial_bytes, size_t tile_bytes) {
    if (partial_bytes > context->scratch_mmq_fixup_bytes ||
        context->scratch_mmq_fixup == nullptr) {
        cudaError_t error = cudaFree(context->scratch_mmq_fixup);
        if (error != cudaSuccess) return error;
        context->scratch_mmq_fixup = nullptr;
        context->scratch_mmq_fixup_bytes = 0;
        error = cudaMalloc(reinterpret_cast<void **>(&context->scratch_mmq_fixup), partial_bytes);
        if (error != cudaSuccess) return error;
        context->scratch_mmq_fixup_bytes = partial_bytes;
    }
    if (tile_bytes > context->scratch_mmq_tiles_bytes || context->scratch_mmq_tiles == nullptr) {
        cudaError_t error = cudaFree(context->scratch_mmq_tiles);
        if (error != cudaSuccess) return error;
        context->scratch_mmq_tiles = nullptr;
        context->scratch_mmq_tiles_bytes = 0;
        error = cudaMalloc(reinterpret_cast<void **>(&context->scratch_mmq_tiles), tile_bytes);
        if (error != cudaSuccess) return error;
        context->scratch_mmq_tiles_bytes = tile_bytes;
    }
    return cudaSuccess;
}

static cudaError_t ensure_token_scratch(fortai_cuda_q8_context_impl *context, size_t bytes) {
    if (bytes <= context->scratch_tokens_bytes && context->scratch_tokens != nullptr)
        return cudaSuccess;
    cudaError_t error = cudaFree(context->scratch_tokens);
    if (error != cudaSuccess) return error;
    context->scratch_tokens = nullptr;
    context->scratch_tokens_bytes = 0;
    error = cudaMalloc(reinterpret_cast<void **>(&context->scratch_tokens), bytes);
    if (error == cudaSuccess) context->scratch_tokens_bytes = bytes;
    return error;
}

static cudaError_t ensure_download_host(fortai_cuda_q8_context_impl *context, size_t bytes) {
    if (bytes <= context->download_host_bytes && context->download_host != nullptr)
        return cudaSuccess;
    /* A resize cannot race an earlier transfer because the public download
     * operation is host-synchronous; still fence the stream for robustness if
     * a larger output is introduced by a future caller. */
    cudaError_t error = cudaStreamSynchronize(context->stream);
    if (error != cudaSuccess) return error;
    if (context->download_host != nullptr) {
        error = cudaFreeHost(context->download_host);
        if (error != cudaSuccess) return error;
    }
    context->download_host = nullptr;
    context->download_host_bytes = 0;
    error = cudaHostAlloc(&context->download_host, bytes, cudaHostAllocPortable);
    if (error == cudaSuccess) context->download_host_bytes = bytes;
    return error;
}

struct host_matvec_request {
    fortai_cuda_q8_weights_impl *weights;
    float *output;
    size_t output_bytes;
};

static void launch_q8_grouped(const host_matvec_request *requests, int count,
    const int8_t *activation, cudaStream_t stream) {
    for (int i = 0; i < count; ++i)
        launch_q8(requests[i].weights, activation, requests[i].output, stream);
}

static int matvec_host_many(fortai_cuda_q8_context_impl *context,
    const host_matvec_request *requests, int count, const void *host_activation,
    size_t activation_bytes, float *elapsed_ms, float *combined_output) {
    if (!context || !requests || count <= 0 || !host_activation || !elapsed_ms)
        return FORTAI_CUDA_INVALID;
    const auto *first = requests[0].weights;
    if (!first || !first->device_data) return FORTAI_CUDA_INVALID;
    const size_t expected_activation = static_cast<size_t>(first->blocks) * q8_block_bytes;
    const bool fused = count <= 4 && first->blocks < 128;
    if (combined_output && !fused) return FORTAI_CUDA_INVALID;
    size_t output_bytes = 0;
    for (int i = 0; i < count; ++i) {
        const auto *weights = requests[i].weights;
        if (!weights || weights->context != context || weights->blocks != first->blocks ||
            !weights->device_data || !requests[i].output) return FORTAI_CUDA_INVALID;
        if (activation_bytes < static_cast<size_t>(weights->blocks) * q8_block_bytes ||
            requests[i].output_bytes < static_cast<size_t>(weights->rows) * sizeof(float))
            return FORTAI_CUDA_INVALID;
        if (fused)
            output_bytes += static_cast<size_t>(weights->rows) * sizeof(float);
        else
            output_bytes = std::max(output_bytes, static_cast<size_t>(weights->rows) * sizeof(float));
    }
    cudaSetDevice(context->device);
    cudaError_t error = ensure_host_matvec_scratch(context, expected_activation, output_bytes);
    if (error == cudaSuccess) error = cudaEventRecord(context->start, context->stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(context->scratch_raw, host_activation,
            expected_activation, cudaMemcpyHostToDevice, context->stream);
    if (error == cudaSuccess) {
        launch_q8_repack_activation(context->scratch_raw, context->scratch_activation,
            first->blocks, context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess && fused) {
        host_matvec_request device_requests[4] = {};
        size_t output_offset = 0;
        for (int i = 0; i < count; ++i) {
            device_requests[i] = requests[i];
            device_requests[i].output = reinterpret_cast<float *>(
                reinterpret_cast<char *>(context->scratch_output) + output_offset);
            output_offset += static_cast<size_t>(requests[i].weights->rows) * sizeof(float);
        }
        launch_q8_grouped(device_requests, count, context->scratch_activation, context->stream);
        error = cudaGetLastError();
        if (combined_output) {
            error = cudaMemcpyAsync(combined_output, context->scratch_output, output_bytes,
                cudaMemcpyDeviceToHost, context->stream);
        } else {
            for (int i = 0; i < count && error == cudaSuccess; ++i)
                error = cudaMemcpyAsync(requests[i].output, device_requests[i].output,
                    static_cast<size_t>(requests[i].weights->rows) * sizeof(float),
                    cudaMemcpyDeviceToHost, context->stream);
        }
    } else {
        for (int i = 0; i < count && error == cudaSuccess; ++i) {
            launch_q8(requests[i].weights, context->scratch_activation,
                context->scratch_output, context->stream);
            error = cudaGetLastError();
            if (error == cudaSuccess)
                error = cudaMemcpyAsync(requests[i].output, context->scratch_output,
                    static_cast<size_t>(requests[i].weights->rows) * sizeof(float),
                    cudaMemcpyDeviceToHost, context->stream);
        }
    }
    if (error == cudaSuccess) error = cudaEventRecord(context->stop, context->stream);
    if (error == cudaSuccess) error = cudaEventSynchronize(context->stop);
    if (error == cudaSuccess) error = cudaEventElapsedTime(elapsed_ms,
        context->start, context->stop);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "grouped host matvec", error);
}

__device__ __forceinline__ float qwen_silu(float value) {
    return value / (1.0f + expf(-value));
}

__device__ __forceinline__ float qwen_sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float qwen_softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return log1pf(expf(value));
}

/* Store one row in the same Q8_0 byte layout used by the GGUF/CPU path:
 * a half precision block scale followed by 32 signed values.  The CUDA
 * resident KV cache deliberately uses this layout so the oracle and the
 * device path have identical rounding and can share the same dequantizer. */
__device__ __forceinline__ void qwen_quantize_cache_block(const float *input,
    int8_t *output, int block) {
    const int input_offset = block * q8_block_width;
    const int output_offset = block * q8_block_bytes;
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < q8_block_width; ++i)
        maximum = fmaxf(maximum, fabsf(input[input_offset + i]));
    const float scale = maximum / 127.0f;
    const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
    const uint16_t scale_bits = __half_as_ushort(__float2half_rn(scale));
    reinterpret_cast<uint16_t *>(output + output_offset)[0] = scale_bits;
#pragma unroll
    for (int i = 0; i < q8_block_width; ++i)
        output[output_offset + 2 + i] = static_cast<int8_t>(__float2int_rn(
            input[input_offset + i] * inverse));
}

/* llama.cpp/GGML Q4_0 layout: one F16 scale and sixteen packed nibbles for
 * each 32-value block.  The signed extremum determines the scale, including
 * its sign; preserving that detail is required for draft-logit parity. */
__device__ __forceinline__ void qwen_quantize_cache_block_q4(const float *input,
    int8_t *output, int block) {
    const int input_offset = block * q4_block_width;
    const int output_offset = block * q4_block_bytes;
    float absolute_maximum = 0.0f;
    float signed_maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < q4_block_width; ++i) {
        const float value = input[input_offset + i];
        if (absolute_maximum < fabsf(value)) {
            absolute_maximum = fabsf(value);
            signed_maximum = value;
        }
    }
    const float scale = signed_maximum / -8.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    reinterpret_cast<uint16_t *>(output + output_offset)[0] =
        __half_as_ushort(__float2half_rn(scale));
#pragma unroll
    for (int i = 0; i < q4_block_width / 2; ++i) {
        const int low = min(15, static_cast<int>(input[input_offset + i] * inverse + 8.5f));
        const int high = min(15, static_cast<int>(
            input[input_offset + q4_block_width / 2 + i] * inverse + 8.5f));
        output[output_offset + 2 + i] = static_cast<int8_t>(low | (high << 4));
    }
}

__device__ __forceinline__ float qwen_load_q4_cache_component(
    const int8_t *row, int component) {
    const int block = component / q4_block_width;
    const int within = component - block * q4_block_width;
    const int packed = static_cast<unsigned char>(
        row[block * q4_block_bytes + 2 + within % (q4_block_width / 2)]);
    const int quant = within < q4_block_width / 2 ? packed & 0x0f : packed >> 4;
    return block_scale(row + block * q4_block_bytes) * static_cast<float>(quant - 8);
}

__global__ void qwen_silu_product(float *gate, const float *up, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    gate[index] = qwen_silu(gate[index]) * up[index];
}

/* Fused gate/up decode MMVQ.  It follows q8_gemv_mmvq's lane
 * ownership while evaluating both resident matrices from the same activation
 * loads.  This preserves the single launch used by the recurrent/FFN fast
 * path and halves the quantized activation traffic relative to two separate
 * GEMVs. */
__launch_bounds__(128, 1)
__global__ void q8_gemv_silu_product_mmvq_four_warp(const int8_t *gate_weights,
    const int8_t *up_weights, const int8_t *activation, float *output,
    int rows, int blocks) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (row >= rows) return;
    const int block_in_warp = lane >> 2;
    const int sublane = lane & 3;
    const int8_t *gate_row = gate_weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    const int8_t *up_row = up_weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    float gate_accumulator = 0.0f;
    float up_accumulator = 0.0f;
    for (int block = warp * 8 + block_in_warp; block < blocks; block += 32) {
        const int8_t *gate_block = gate_row + block * q8_block_bytes;
        const int8_t *up_block = up_row + block * q8_block_bytes;
        const int8_t *activation_block = activation + block * q8_activation_block_bytes;
        const int word = sublane * 2;
        const int8_t *activation_words = activation_block + q8_activation_data_offset;
        int gate_dot = __dp4a(load_i32(gate_block + 2 + word * 4),
            load_i32(activation_words + word * 4), 0);
        int up_dot = __dp4a(load_i32(up_block + 2 + word * 4),
            load_i32(activation_words + word * 4), 0);
        gate_dot = __dp4a(load_i32(gate_block + 2 + (word + 1) * 4),
            load_i32(activation_words + (word + 1) * 4), gate_dot);
        up_dot = __dp4a(load_i32(up_block + 2 + (word + 1) * 4),
            load_i32(activation_words + (word + 1) * 4), up_dot);
        const float activation_scale = block_scale(activation_block);
        gate_accumulator = fmaf(block_scale(gate_block) * activation_scale,
            static_cast<float>(gate_dot), gate_accumulator);
        up_accumulator = fmaf(block_scale(up_block) * activation_scale,
            static_cast<float>(up_dot), up_accumulator);
    }
    const unsigned mask = 0xffffffffu;
    for (int offset = 2; offset > 0; offset >>= 1) {
        gate_accumulator += __shfl_down_sync(mask, gate_accumulator, offset, 4);
        up_accumulator += __shfl_down_sync(mask, up_accumulator, offset, 4);
    }
    __shared__ float gate_partial[32];
    __shared__ float up_partial[32];
    if (sublane == 0) {
        gate_partial[warp * 8 + block_in_warp] = gate_accumulator;
        up_partial[warp * 8 + block_in_warp] = up_accumulator;
    }
    __syncthreads();
    if (warp == 0) {
        gate_accumulator = gate_partial[lane];
        up_accumulator = up_partial[lane];
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_accumulator += __shfl_down_sync(0xffffffffu, gate_accumulator, offset);
            up_accumulator += __shfl_down_sync(0xffffffffu, up_accumulator, offset);
        }
        if (lane == 0) output[row] = qwen_silu(gate_accumulator) * up_accumulator;
    }
}

__global__ void q8_gemv_silu_product_one_warp(const int8_t *gate_weights,
    const int8_t *up_weights, const int8_t *activation, float *output,
    int rows, int blocks) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows) return;
    const int8_t *gate_row = gate_weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    const int8_t *up_row = up_weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    float gate_accumulator = 0.0f;
    float up_accumulator = 0.0f;
    if (lane < 8) for (int block = 0; block < blocks; ++block) {
        const int8_t *gate_block = gate_row + block * q8_block_bytes;
        const int8_t *up_block = up_row + block * q8_block_bytes;
        const int8_t *activation_block = activation + block * q8_activation_block_bytes;
        const float gate_scale = block_scale(gate_block) * block_scale(activation_block);
        const float up_scale = block_scale(up_block) * block_scale(activation_block);
        const int group = lane * 4;
        const int activation_values = load_i32(activation_block + q8_activation_data_offset + group);
        const int gate_dot = __dp4a(load_i32(gate_block + 2 + group), activation_values, 0);
        const int up_dot = __dp4a(load_i32(up_block + 2 + group), activation_values, 0);
        gate_accumulator = fmaf(gate_scale, static_cast<float>(gate_dot), gate_accumulator);
        up_accumulator = fmaf(up_scale, static_cast<float>(up_dot), up_accumulator);
    }
    for (int offset = 4; offset; offset >>= 1) {
        gate_accumulator += __shfl_down_sync(0xffffffffu, gate_accumulator, offset);
        up_accumulator += __shfl_down_sync(0xffffffffu, up_accumulator, offset);
    }
    if (lane == 0) output[row] = qwen_silu(gate_accumulator) * up_accumulator;
}

static void launch_q8_silu_product(fortai_cuda_q8_weights_impl *gate,
    fortai_cuda_q8_weights_impl *up, const int8_t *activation, float *output,
    cudaStream_t stream) {
    if (gate->blocks < 32) {
        q8_gemv_silu_product_one_warp<<<gate->rows, 32, 0, stream>>>(gate->device_data,
            up->device_data, activation, output, gate->rows, gate->blocks);
    } else {
        /* llama.cpp fuses the two MMVQ projections and SwiGLU for every
         * scalar decode shape, not only the small-block case.  The fused
         * kernel keeps one activation load per Q8 block and one launch for
         * gate/up, which is especially important for the 128-block Qwen
         * FFN matrices used by the 27B model. */
        q8_gemv_silu_product_mmvq_four_warp<<<gate->rows, 128, 0, stream>>>(gate->device_data,
            up->device_data, activation, output, gate->rows, gate->blocks);
    }
}

__global__ void qwen_recurrent_conv_silu(float *qkv, const float *conv_weights,
    float *conv_state, int conv_size, int conv_kernel) {
    const int channel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= conv_size) return;
    float accumulator = 0.0f;
    for (int slot = 0; slot < conv_kernel; ++slot)
        accumulator += conv_weights[channel * conv_kernel + slot] *
            (slot == conv_kernel - 1 ? qkv[channel] :
                conv_state[slot * conv_size + channel]);
    for (int slot = 0; slot < conv_kernel - 2; ++slot)
        conv_state[slot * conv_size + channel] = conv_state[(slot + 1) * conv_size + channel];
    conv_state[(conv_kernel - 2) * conv_size + channel] = qkv[channel];
    qkv[channel] = qwen_silu(accumulator);
}

__global__ void qwen_recurrent_l2_normalize(float *values, int slices, int length,
    float epsilon) {
    const int slice = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (slice >= slices) return;
    __shared__ float partial[4];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    float sum = 0.0f;
    for (int i = tid; i < length; i += blockDim.x) {
        const float value = values[slice * length + i];
        sum = fmaf(value, value, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    sum = lane < 4 ? partial[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) partial[0] = sum;
    }
    __syncthreads();
    float norm = sqrtf(partial[0]);
    if (norm < epsilon) norm = epsilon;
    const float inverse = 1.0f / norm;
    for (int i = tid; i < length; i += blockDim.x)
        values[slice * length + i] *= inverse;
}

/* One block owns four rows of one GDN head.  Each warp carries one complete
 * row, so the delta update, state write, and output dot product are completed
 * in one launch without the old one-block-per-row scheduling overhead.  The
 * row-major state layout is intentional: it is the layout used by the CPU
 * oracle and by the public state inspection helpers. */
__global__ void qwen_recurrent_gdn_fused(const float *qkv, const float *alpha,
    const float *beta, const float *ssm_a, const float *ssm_dt,
    float *state, float *output, int state_size, int key_heads, int value_heads,
    int head_size) {
    const int row_index = static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const int head = row_index / head_size;
    const int row = row_index % head_size;
    if (head >= value_heads || row >= head_size || lane >= 32) return;
    const int key_head = head % key_heads;
    const int q_offset = key_head * state_size;
    const int k_offset = state_size * key_heads + key_head * state_size;
    const int v_offset = 2 * state_size * key_heads + head * head_size;
    const float decay = expf(ssm_a[head] * qwen_softplus(alpha[head] + ssm_dt[head]));
    const float beta_value = qwen_sigmoid(beta[head]);
    float *state_row = state + static_cast<size_t>(head) * head_size * head_size +
        static_cast<size_t>(row) * head_size;
    float key_dot = 0.0f;
    for (int col = lane; col < head_size; col += 32) {
        const float entry = state_row[col] * decay;
        state_row[col] = entry;
        key_dot = fmaf(entry, qkv[k_offset + col], key_dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        key_dot += __shfl_down_sync(0xffffffffu, key_dot, offset);
    const float delta = (qkv[v_offset + row] - key_dot) * beta_value;
    const float shared_delta = __shfl_sync(0xffffffffu, delta, 0);
    float query_dot = 0.0f;
    for (int col = lane; col < head_size; col += 32) {
        const float entry = state_row[col] + shared_delta * qkv[k_offset + col];
        state_row[col] = entry;
        query_dot = fmaf(entry, qkv[q_offset + col], query_dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        query_dot += __shfl_down_sync(0xffffffffu, query_dot, offset);
    if (lane == 0)
        output[row_index] = query_dot * rsqrtf(static_cast<float>(head_size));
}

/* The production Qwen3.5 model uses a 128-wide recurrent head.  Keep a
 * shape-specialized launch alongside the metadata-driven kernel above: the
 * 3-D grid makes the head/column coordinates free, and the compile-time
 * width lets nvcc unroll the four row lanes exactly like llama.cpp's
 * gated_delta_net kernel.  The device state remains row-major (the same
 * layout as the CPU oracle), so this is an execution-only optimization. */
template <int HeadSize>
__global__ void qwen_recurrent_gdn_fused_fixed(const float *qkv, const float *alpha,
    const float *beta, const float *ssm_a, const float *ssm_dt,
    float *state, float *output, int state_size, int key_heads, int value_heads) {
    constexpr int warp_size = 32;
    constexpr int rows_per_lane = HeadSize / warp_size;
    const int head = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.z * blockDim.y + threadIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (head >= value_heads || column >= HeadSize || lane >= warp_size) return;

    const int key_head = head % key_heads;
    const int q_offset = key_head * state_size;
    const int k_offset = state_size * key_heads + key_head * state_size;
    const int v_offset = 2 * state_size * key_heads + head * HeadSize;
    const float decay = expf(ssm_a[head] * qwen_softplus(alpha[head] + ssm_dt[head]));
    const float beta_value = qwen_sigmoid(beta[head]);
    float *state_row = state + static_cast<size_t>(head) * HeadSize * HeadSize +
        static_cast<size_t>(column) * HeadSize;
    float state_shard[rows_per_lane];

#pragma unroll
    for (int row = 0; row < rows_per_lane; ++row) {
        const int index = row * warp_size + lane;
        state_shard[row] = state_row[index];
    }

    float key_dot = 0.0f;
#pragma unroll
    for (int row = 0; row < rows_per_lane; ++row) {
        const int index = row * warp_size + lane;
        /* Form the decayed state before both reductions, matching the
         * resident scalar path's established state-update order. */
        state_shard[row] = __fmul_rn(state_shard[row], decay);
        key_dot = fmaf(state_shard[row], qkv[k_offset + index], key_dot);
    }
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        key_dot += __shfl_down_sync(0xffffffffu, key_dot, offset);
    const float delta = (qkv[v_offset + column] - key_dot) * beta_value;
    const float shared_delta = __shfl_sync(0xffffffffu, delta, 0);

    float query_dot = 0.0f;
#pragma unroll
    for (int row = 0; row < rows_per_lane; ++row) {
        const int index = row * warp_size + lane;
        state_shard[row] = fmaf(shared_delta, qkv[k_offset + index], state_shard[row]);
        query_dot = fmaf(state_shard[row], qkv[q_offset + index], query_dot);
    }
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        query_dot += __shfl_down_sync(0xffffffffu, query_dot, offset);
    if (lane == 0)
        output[head * HeadSize + column] = query_dot * rsqrtf(static_cast<float>(HeadSize));

#pragma unroll
    for (int row = 0; row < rows_per_lane; ++row) {
        const int index = row * warp_size + lane;
        state_row[index] = state_shard[row];
    }
}

static void launch_qwen_recurrent_gdn_fused(const float *qkv, const float *alpha,
    const float *beta, const float *ssm_a, const float *ssm_dt,
    float *state, float *output, int state_size, int key_heads, int value_heads,
    int head_size, cudaStream_t stream) {
    if (state_size == 128 && head_size == 128) {
        qwen_recurrent_gdn_fused_fixed<128><<<
            dim3(static_cast<unsigned>(value_heads), 1, 32), dim3(32, 4, 1), 0, stream>>>(
            qkv, alpha, beta, ssm_a, ssm_dt, state, output, state_size, key_heads, value_heads);
    } else {
        qwen_recurrent_gdn_fused<<<
            (value_heads * head_size + 3) / 4, dim3(32, 4, 1), 0, stream>>>(
            qkv, alpha, beta, ssm_a, ssm_dt, state, output, state_size,
            key_heads, value_heads, head_size);
    }
}

__global__ void qwen_recurrent_conv_silu_batch(float *qkv, const float *conv_weights,
    float *conv_state, float *conv_state_first, float *conv_state_second,
    int conv_size, int conv_kernel, int batch) {
    const int channel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= conv_size) return;
    /* A single channel owns its short convolution history.  The token loop is
     * serialized per channel (the state dependency is real), while all
     * channels and all prompt columns run concurrently. */
    for (int token = 0; token < batch; ++token) {
        float accumulator = 0.0f;
        const size_t token_offset = static_cast<size_t>(token) * conv_size + channel;
        for (int slot = 0; slot < conv_kernel; ++slot)
            accumulator += conv_weights[channel * conv_kernel + slot] *
                (slot == conv_kernel - 1 ? qkv[token_offset] :
                    conv_state[slot * conv_size + channel]);
        for (int slot = 0; slot < conv_kernel - 2; ++slot)
            conv_state[slot * conv_size + channel] =
                conv_state[(slot + 1) * conv_size + channel];
        conv_state[(conv_kernel - 2) * conv_size + channel] = qkv[token_offset];
        if (token == 0) {
            for (int slot = 0; slot < conv_kernel - 1; ++slot)
                conv_state_first[slot * conv_size + channel] =
                    conv_state[slot * conv_size + channel];
        }
        if (token == 1) {
            for (int slot = 0; slot < conv_kernel - 1; ++slot)
                conv_state_second[slot * conv_size + channel] =
                    conv_state[slot * conv_size + channel];
        }
        qkv[token_offset] = qwen_silu(accumulator);
    }
}

__global__ void qwen_recurrent_l2_normalize_batch(float *values, int slices,
    int length, int batch, int stride, float epsilon) {
    const int slice = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (slice >= slices || token >= batch) return;
    __shared__ float partial[32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const size_t base = static_cast<size_t>(token) * stride +
        static_cast<size_t>(slice) * length;
    float sum = 0.0f;
    for (int i = tid; i < length; i += blockDim.x)
        sum = fmaf(values[base + i], values[base + i], sum);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < (blockDim.x + 31) / 32 ? partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) partial[0] = sum;
    }
    __syncthreads();
    /* This is L2 normalization, not RMS normalization: the scalar oracle
     * divides by sqrt(sum(x*x)) without a length factor.  Keeping that
     * distinction is essential because the Qwen recurrent heads are later
     * projected with a fixed 1/sqrt(head_size) scale. */
    const float norm = sqrtf(partial[0]);
    const float inverse = norm > epsilon ? 1.0f / norm : 1.0f / epsilon;
    for (int i = tid; i < length; i += blockDim.x)
        values[base + i] *= inverse;
}

/* Sequence-batched GDN with the same row/warp ownership as the scalar
 * kernels.  The token loop is the only true dependency; keeping it inside a
 * head block preserves both the recurrent state order and the scalar dot
 * reduction order while all heads and rows remain resident on the device. */
template <int HeadSize>
__global__ void qwen_recurrent_gdn_fused_batch_fixed(const float *__restrict__ qkv,
    const float *__restrict__ alpha, const float *__restrict__ beta,
    const float *__restrict__ ssm_a, const float *__restrict__ ssm_dt,
    float *__restrict__ state, float *__restrict__ state_first,
    float *__restrict__ state_second,
    float *__restrict__ output, int state_size,
    int key_heads, int value_heads, int inner_size, int batch) {
    constexpr int warp_size = 32;
    constexpr int rows_per_lane = HeadSize / warp_size;
    const int head = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.z * blockDim.y + threadIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (head >= value_heads || column >= HeadSize || lane >= warp_size) return;
    const int key_head = head % key_heads;
    const int q_offset = key_head * state_size;
    const int k_offset = state_size * key_heads + key_head * state_size;
    const int v_offset = 2 * state_size * key_heads + head * HeadSize;
    float *state_row = state + static_cast<size_t>(head) * HeadSize * HeadSize +
        static_cast<size_t>(column) * HeadSize;
    float state_shard[rows_per_lane];
    float key_shard[rows_per_lane];
    float query_shard[rows_per_lane];

#pragma unroll
    for (int row = 0; row < rows_per_lane; ++row) {
        const int index = row * warp_size + lane;
        state_shard[row] = state_row[index];
    }

    for (int token = 0; token < batch; ++token) {
        const size_t qkv_base = static_cast<size_t>(token) * (2 * state_size * key_heads +
            value_heads * HeadSize);
        const float decay = expf(ssm_a[head] * qwen_softplus(
            alpha[static_cast<size_t>(token) * value_heads + head] + ssm_dt[head]));
        const float beta_value = qwen_sigmoid(
            beta[static_cast<size_t>(token) * value_heads + head]);
        float key_dot = 0.0f;
#pragma unroll
        for (int row = 0; row < rows_per_lane; ++row) {
            const int index = row * warp_size + lane;
            key_shard[row] = qkv[qkv_base + k_offset + index];
            query_shard[row] = qkv[qkv_base + q_offset + index];
            state_shard[row] = __fmul_rn(state_shard[row], decay);
            key_dot = fmaf(state_shard[row], key_shard[row], key_dot);
        }
#pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset >>= 1)
            key_dot += __shfl_down_sync(0xffffffffu, key_dot, offset);
        const float delta = (qkv[qkv_base + v_offset + column] - key_dot) * beta_value;
        const float shared_delta = __shfl_sync(0xffffffffu, delta, 0);
        float query_dot = 0.0f;
#pragma unroll
        for (int row = 0; row < rows_per_lane; ++row) {
            state_shard[row] = fmaf(shared_delta, key_shard[row], state_shard[row]);
            query_dot = fmaf(state_shard[row], query_shard[row], query_dot);
        }
#pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset >>= 1)
            query_dot += __shfl_down_sync(0xffffffffu, query_dot, offset);
        if (lane == 0)
            output[static_cast<size_t>(token) * inner_size + head * HeadSize + column] =
                query_dot * rsqrtf(static_cast<float>(HeadSize));
        if (token == 0) {
#pragma unroll
            for (int row = 0; row < rows_per_lane; ++row) {
                const int index = row * warp_size + lane;
                state_first[static_cast<size_t>(head) * HeadSize * HeadSize +
                    static_cast<size_t>(column) * HeadSize + index] = state_shard[row];
            }
        }
        if (token == 1) {
#pragma unroll
            for (int row = 0; row < rows_per_lane; ++row) {
                const int index = row * warp_size + lane;
                state_second[static_cast<size_t>(head) * HeadSize * HeadSize +
                    static_cast<size_t>(column) * HeadSize + index] = state_shard[row];
            }
        }
    }
#pragma unroll
    for (int row = 0; row < rows_per_lane; ++row) {
        const int index = row * warp_size + lane;
        state_row[index] = state_shard[row];
    }
}

__global__ void qwen_recurrent_gdn_fused_batch_generic(const float *qkv,
    const float *alpha, const float *beta, const float *ssm_a,
    const float *ssm_dt, float *state, float *state_first, float *state_second,
    float *output, int state_size,
    int key_heads, int value_heads, int head_size, int inner_size, int batch) {
    const int row_index = static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const int head = row_index / head_size;
    const int row = row_index % head_size;
    if (head >= value_heads || row >= head_size || lane >= 32) return;
    const int key_head = head % key_heads;
    const int q_offset = key_head * state_size;
    const int k_offset = state_size * key_heads + key_head * state_size;
    const int v_offset = 2 * state_size * key_heads + head * head_size;
    float *state_row = state + static_cast<size_t>(head) * head_size * head_size +
        static_cast<size_t>(row) * head_size;
    for (int token = 0; token < batch; ++token) {
        const size_t qkv_base = static_cast<size_t>(token) * (2 * state_size * key_heads +
            value_heads * head_size);
        const float decay = expf(ssm_a[head] * qwen_softplus(
            alpha[static_cast<size_t>(token) * value_heads + head] + ssm_dt[head]));
        const float beta_value = qwen_sigmoid(
            beta[static_cast<size_t>(token) * value_heads + head]);
        float key_dot = 0.0f;
        for (int col = lane; col < head_size; col += 32) {
            const float entry = state_row[col] * decay;
            state_row[col] = entry;
            key_dot = fmaf(entry, qkv[qkv_base + k_offset + col], key_dot);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            key_dot += __shfl_down_sync(0xffffffffu, key_dot, offset);
        const float delta = (qkv[qkv_base + v_offset + row] - key_dot) * beta_value;
        const float shared_delta = __shfl_sync(0xffffffffu, delta, 0);
        float query_dot = 0.0f;
        for (int col = lane; col < head_size; col += 32) {
            const float entry = state_row[col] + shared_delta * qkv[qkv_base + k_offset + col];
            state_row[col] = entry;
            query_dot = fmaf(entry, qkv[qkv_base + q_offset + col], query_dot);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            query_dot += __shfl_down_sync(0xffffffffu, query_dot, offset);
        if (lane == 0)
            output[static_cast<size_t>(token) * inner_size + head * head_size + row] =
                query_dot * rsqrtf(static_cast<float>(head_size));
        if (token == 0) {
            float *first_row = state_first + static_cast<size_t>(head) * head_size * head_size +
                static_cast<size_t>(row) * head_size;
            for (int col = lane; col < head_size; col += 32)
                first_row[col] = state_row[col];
        }
        if (token == 1) {
            float *second_row = state_second + static_cast<size_t>(head) * head_size * head_size +
                static_cast<size_t>(row) * head_size;
            for (int col = lane; col < head_size; col += 32)
                second_row[col] = state_row[col];
        }
    }
}

__global__ void qwen_recurrent_gdn_norm_gate_batch(float *output,
    const float *gate, const float *ssm_norm, int value_heads, int head_size,
    int inner_size, int batch, float norm_epsilon) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (head >= value_heads || token >= batch) return;
    __shared__ float partial[4];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const size_t base = static_cast<size_t>(token) * inner_size + head * head_size;
    const float value = tid < head_size ? output[base + tid] : 0.0f;
    float sum = value * value;
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 4 ? partial[lane] : 0.0f;
        for (int offset = 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) partial[0] = sum;
    }
    __syncthreads();
    const float inverse = rsqrtf(partial[0] / static_cast<float>(head_size) + norm_epsilon);
    if (tid < head_size)
        output[base + tid] = value * inverse * ssm_norm[tid] *
            qwen_silu(gate[base + tid]);
}

__global__ void qwen_recurrent_gdn_norm_gate(float *output, const float *gate,
    const float *ssm_norm, int value_heads, int head_size, float norm_epsilon) {
    const int head = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (head >= value_heads) return;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float partial[4];
    const size_t base = static_cast<size_t>(head) * head_size;
    const float value = tid < head_size ? output[base + tid] : 0.0f;
    float sum = value * value;
    for (int delta = 16; delta > 0; delta >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, delta);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    sum = lane < 4 ? partial[lane] : 0.0f;
    if (warp == 0) {
        for (int delta = 16; delta > 0; delta >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, delta);
        if (lane == 0) partial[0] = sum;
    }
    __syncthreads();
    const float inverse = rsqrtf(partial[0] / static_cast<float>(head_size) + norm_epsilon);
    if (tid < head_size)
        output[base + tid] = value * inverse * ssm_norm[tid] * qwen_silu(gate[base + tid]);
}

__global__ void qwen_quantize_q8(const float *input, int8_t *output, int blocks) {
    const int block = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (block >= blocks) return;
    const int input_offset = block * q8_block_width;
    const int output_offset = block * q8_activation_block_bytes;
    float maximum = 0.0f;
    for (int i = 0; i < q8_block_width; ++i)
        maximum = fmaxf(maximum, fabsf(input[input_offset + i]));
    const float scale = maximum / 127.0f;
    const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
    const uint16_t scale_bits = __half_as_ushort(__float2half_rn(scale));
    reinterpret_cast<uint16_t *>(output + output_offset)[0] = scale_bits;
    reinterpret_cast<uint16_t *>(output + output_offset + 2)[0] = 0;
    for (int i = 0; i < q8_block_width; ++i)
        output[output_offset + q8_activation_data_offset + i] = static_cast<int8_t>(__float2int_rn(
        input[input_offset + i] * inverse));
}

/* Quantize all columns of a prompt matrix.  The token/column is the second
 * grid dimension, so every block works on one independent 32-value Q8
 * block and all columns can be prepared in parallel. */
__global__ void qwen_quantize_q8_batch(const float *input, int8_t *output,
    int width, int blocks, int batch) {
    const int block = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (block >= blocks || token >= batch || lane >= q8_block_width) return;
    const size_t input_offset = static_cast<size_t>(token) * width +
        static_cast<size_t>(block) * q8_block_width;
    const size_t output_offset = (static_cast<size_t>(token) * blocks + block) *
        q8_activation_block_bytes;
    __shared__ float partial[32];
    const float value = input[input_offset + lane];
    partial[lane] = fabsf(value);
    __syncthreads();
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] = fmaxf(partial[lane], partial[lane + stride]);
        __syncthreads();
    }
    const float maximum = partial[0];
    const float scale = maximum / 127.0f;
    const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
    if (lane == 0) {
        const uint16_t scale_bits = __half_as_ushort(__float2half_rn(scale));
        reinterpret_cast<uint16_t *>(output + output_offset)[0] = scale_bits;
        reinterpret_cast<uint16_t *>(output + output_offset + 2)[0] = 0;
    }
    output[output_offset + q8_activation_data_offset + lane] =
        static_cast<int8_t>(__float2int_rn(value * inverse));
}

/* Quantization layout consumed by the MMQ prompt kernel.  Four consecutive
 * Q8 blocks are packed as one 128-value row: four float scales followed by
 * 128 int8 values.  Rows are group-major, matching llama.cpp's
 * block_q8_1_mmq layout so a whole token tile can be copied coalesced into
 * shared memory.  The byte size is identical to the ordinary four 36-byte
 * activation blocks, so the existing scratch allocation is reused. */
__global__ void qwen_quantize_q8_mmq_batch(const float *input, int8_t *output,
    int width, int groups, int batch) {
    const int group = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (group >= groups || token >= batch || lane >= 128) return;
    const int block = lane >> 5;
    const int element = lane & 31;
    const size_t input_offset = static_cast<size_t>(token) * width +
        static_cast<size_t>(group) * 128 + block * q8_block_width;
    const float value = input[input_offset + element];
    float maximum = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset, 32));
    const float scale = __half2float(__float2half_rn(maximum / 127.0f));
    const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
    const size_t output_offset = (static_cast<size_t>(group) * batch + token) * 144;
    if (element == 0)
        reinterpret_cast<float *>(output + output_offset)[block] = scale;
    output[output_offset + 16 + block * q8_block_width + element] =
        static_cast<int8_t>(__float2int_rn(value * inverse));
}

/* One block computes one output row for one prompt column.  Four warps
 * cooperate over the quantized blocks; the layout mirrors the MMVQ scalar
 * decode kernel
 * while exposing the token dimension to the grid, turning a sequence of
 * GEMVs into a proper GEMM without an intermediate host round trip. */
__global__ void q8_gemm_batch(const int8_t *__restrict__ weights,
    const int8_t *__restrict__ activation, int rows, int blocks, int batch,
    float *__restrict__ output) {
    const int row = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (row >= rows || token >= batch) return;
    const int8_t *row_weights = weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    const int8_t *token_activation = activation + static_cast<size_t>(token) * blocks * q8_activation_block_bytes;
    float accumulator = 0.0f;
    for (int block = warp * 32 + lane; block < blocks; block += 128) {
        const int8_t *weight_block = row_weights + block * q8_block_bytes;
        const int8_t *activation_block = token_activation + block * q8_activation_block_bytes;
        const float scale = block_scale(weight_block) * block_scale(activation_block);
        int dot = 0;
#pragma unroll
        for (int i = 0; i < q8_block_width; i += 4)
            dot = __dp4a(load_i32(weight_block + 2 + i),
                load_i32(activation_block + q8_activation_data_offset + i), dot);
        accumulator = fmaf(scale, static_cast<float>(dot), accumulator);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    __shared__ float partial[4];
    if (lane == 0) partial[warp] = accumulator;
    __syncthreads();
    if (warp == 0) {
        const int active_warps = (blockDim.x + 31) / 32;
        accumulator = lane < active_warps ? partial[lane] : 0.0f;
        for (int offset = 16; offset >= 1; offset >>= 1)
            accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
        if (lane == 0) output[static_cast<size_t>(token) * rows + row] = accumulator;
    }
}

/* Q8_0 MMQ layout.  This is the compact specialization of llama.cpp's
 * dp4a MMQ path for FortAI's resident Q8_0 weights.  The x tile stores eight
 * 32-value weight blocks per row (with one padding word per row); the y tile
 * stores four activation blocks per token.  A warp owns sixteen rows and all
 * token columns, so every lane computes the same 32 complete outputs as the
 * reference MMQ kernel while each block's scale is applied independently. */
template <int RowsPerTile = 128, int TokensPerTile = 64, int BlocksPerTile = 8,
    bool FullTiles = false>
__launch_bounds__(256, 1)
__global__ void q8_gemm_batch_mmq_layout(const int8_t *__restrict__ weights,
    const int8_t *__restrict__ activation, int rows, int blocks, int batch,
    float *__restrict__ output, float *__restrict__ tmp_fixup,
    int32_t *__restrict__ tile_ids) {
    constexpr int nwarps = 8;
    constexpr int x_q_stride = 2 * q8_block_width + 1;
    constexpr int x_scale_stride = BlocksPerTile;
    constexpr int x_q_words = RowsPerTile * x_q_stride;
    constexpr int x_scale_words = RowsPerTile * x_scale_stride + RowsPerTile / 4;
    constexpr int y_stride = 4 + 4 * q8_block_width / 4;
    constexpr int shared_x_scale_offset = x_q_words * static_cast<int>(sizeof(int));
    constexpr int shared_y_offset = shared_x_scale_offset +
        x_scale_words * static_cast<int>(sizeof(float));
    extern __shared__ unsigned char tile_storage[];
    auto *x_q = reinterpret_cast<int *>(tile_storage);
    auto *x_scale = reinterpret_cast<float *>(tile_storage + shared_x_scale_offset);
    auto *y = reinterpret_cast<int *>(tile_storage + shared_y_offset);
    auto *y_scale = reinterpret_cast<float *>(y);

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int linear_tid = warp * 32 + lane;
    const int row_tiles = (rows + RowsPerTile - 1) / RowsPerTile;
    const int total_tiles = row_tiles * ((batch + TokensPerTile - 1) / TokensPerTile);
    const int k_tiles = (blocks + BlocksPerTile - 1) / BlocksPerTile;
    const int total_units = total_tiles * k_tiles;
    (void) tmp_fixup;
    (void) tile_ids;

    for (int unit = static_cast<int>(blockIdx.x); unit < total_units;
        unit += static_cast<int>(gridDim.x)) {
        const int tile_index = unit / k_tiles;
        const int row_tile = (tile_index % row_tiles) * RowsPerTile;
        const int token_tile = (tile_index / row_tiles) * TokensPerTile;
        float sum[TokensPerTile * RowsPerTile / (nwarps * 32)] = {};
        const int k_base = (unit % k_tiles) * BlocksPerTile;

#pragma unroll
        for (int i0 = 0; i0 < RowsPerTile; i0 += nwarps) {
            const int row = i0 + warp;
            const int block = lane / 8;
            const int word = lane & 7;
            const int source_row = row_tile + row;
            const int source_block = k_base + block;
            const bool valid = FullTiles || (source_row < rows && source_block < blocks);
            const int8_t *source = valid ? weights +
                (static_cast<size_t>(source_row) * blocks + source_block) * q8_block_bytes + 2 : nullptr;
            x_q[row * x_q_stride + lane] = valid ? load_i32(source + word * 4) : 0;
            const int second_block = k_base + block + 4;
            const bool second_valid = FullTiles ||
                (source_row < rows && second_block < blocks);
            const int8_t *second_source = second_valid ? weights +
                (static_cast<size_t>(source_row) * blocks + second_block) * q8_block_bytes + 2 : nullptr;
            x_q[row * x_q_stride + q8_block_width + lane] = second_valid ?
                load_i32(second_source + word * 4) : 0;
        }
#pragma unroll
        for (int i0 = 0; i0 < RowsPerTile; i0 += nwarps * 4) {
            const int row = i0 + warp * 4 + lane / 8;
            const int block = lane & 7;
            const int source_block = k_base + block;
            const bool valid = FullTiles ||
                (row_tile + row < rows && source_block < blocks);
            x_scale[row * x_scale_stride + row / 4 + block] = valid ? block_scale(
                weights + (static_cast<size_t>(row_tile + row) * blocks + source_block) *
                    q8_block_bytes) : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int group = 0; group < 2; ++group) {
            const int activation_group = k_base / 4 + group;
            for (int index = linear_tid; index < TokensPerTile * y_stride;
                index += nwarps * 32) {
                const int token = index / y_stride;
                const int word = index % y_stride;
                const int source_token = token_tile + token;
                const bool valid = FullTiles || (source_token < batch &&
                    activation_group < (blocks + 3) / 4);
                y[token * y_stride + word] = valid ? reinterpret_cast<const int *>(activation)[
                    (static_cast<size_t>(activation_group) * batch + source_token) * y_stride +
                    word] : 0;
            }
            __syncthreads();

#pragma unroll
            for (int j0 = 0; j0 < TokensPerTile; j0 += nwarps) {
                const int token = j0 + warp;
#pragma unroll
                for (int i0 = 0; i0 < RowsPerTile; i0 += 32) {
                    const int row = i0 + lane;
                    const int output_index = (j0 / nwarps) * (RowsPerTile / 32) + i0 / 32;
                    if (FullTiles || (token_tile + token < batch && row_tile + row < rows)) {
                        float accumulator = sum[output_index];
#pragma unroll
                        for (int block = 0; block < 4; ++block) {
                            const int dot_offset = block * 8;
                            int dot = 0;
#pragma unroll
                            for (int word = 0; word < 8; ++word)
                                dot = __dp4a(x_q[row * x_q_stride + group * q8_block_width +
                                    dot_offset + word], y[token * y_stride + 4 + dot_offset + word], dot);
                            accumulator = fmaf(x_scale[row * x_scale_stride + row / 4 +
                                group * 4 + block] * y_scale[token * y_stride + block],
                                static_cast<float>(dot), accumulator);
                        }
                        sum[output_index] = accumulator;
                    }
                }
            }
            __syncthreads();
        }

        const bool tile_complete = k_tiles == 1;
#pragma unroll
        for (int j0 = 0; j0 < TokensPerTile; j0 += nwarps) {
            const int token = j0 + warp;
#pragma unroll
            for (int i0 = 0; i0 < RowsPerTile; i0 += 32) {
                const int row = i0 + lane;
                const int output_index = (j0 / nwarps) * (RowsPerTile / 32) + i0 / 32;
                if (token_tile + token < batch && row_tile + row < rows) {
                    const size_t output_index_global = static_cast<size_t>(token_tile + token) * rows +
                        row_tile + row;
                    if (tile_complete)
                        output[output_index_global] = sum[output_index];
                    else
                        atomicAdd(output + output_index_global, sum[output_index]);
                }
            }
        }
        __syncthreads();
    }
}

/* Minimal native MMA fragments.  This is the same PTX contract used by
 * llama.cpp's Q8 MMQ kernel, kept local so FortAI has no dependency on its
 * headers or runtime.  The shared-memory layout is x[128][76] ints and
 * y[64][36] ints, with the first four y words holding the four float scales. */
struct fortai_mma_tile_a { int x[4]; };
struct fortai_mma_tile_b { int x[2]; };
struct fortai_mma_tile_c { int x[4]; };

__device__ __forceinline__ void fortai_mma_load_a(fortai_mma_tile_a &tile,
    const int *source, int stride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const int *address = source + (static_cast<int>(threadIdx.x) % 16) * stride +
        (static_cast<int>(threadIdx.x) / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(tile.x[0]), "=r"(tile.x[1]), "=r"(tile.x[2]), "=r"(tile.x[3])
        : "l"(address));
#else
    const int lane = static_cast<int>(threadIdx.x);
    const int row = lane % 16;
    const int column = (lane / 16) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        tile.x[i] = source[row * stride + column + i];
#endif
}

/* On NVIDIA, llama.cpp deliberately uses scalar/generic loads for the Q8
 * right-hand tile: the 8x8 B fragment is already laid out in the warp's
 * natural row/column order, and ldmatrix adds an unnecessary transpose path.
 * Keep the same two-int-per-lane mapping here. */
__device__ __forceinline__ void fortai_mma_load_b_generic(fortai_mma_tile_b &tile,
    const int *source, int stride) {
    const int lane = static_cast<int>(threadIdx.x);
    const int row = lane / 4;
    const int column = lane & 3;
    tile.x[0] = source[row * stride + column];
    tile.x[1] = source[row * stride + 4 + column];
}

__device__ __forceinline__ void fortai_mma_i8(fortai_mma_tile_c &destination,
    const fortai_mma_tile_a &left, const fortai_mma_tile_b &right) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+r"(destination.x[0]), "+r"(destination.x[1]),
          "+r"(destination.x[2]), "+r"(destination.x[3])
        : "r"(left.x[0]), "r"(left.x[1]), "r"(left.x[2]), "r"(left.x[3]),
          "r"(right.x[0]), "r"(right.x[1]));
#else
    /* The host dispatch only selects this kernel on MMA-capable devices. */
    (void) destination;
    (void) left;
    (void) right;
#endif
}

/* Keep each accumulator index a compile-time constant.  The tensor-core
 * fragment gives one thread four rows by two columns; spelling those four
 * updates out avoids a dynamically indexed local array and the local-memory
 * spills that otherwise dominate the wide-token kernel. */
template <int J0, int N>
__device__ __forceinline__ void fortai_mma_accumulate_product(
    const fortai_mma_tile_a &left, const fortai_mma_tile_b &right,
    const float *x_df, const float *y_df, int x_stride, int y_stride,
    int group, int k01, int warp, int lane, float *sum) {
    fortai_mma_tile_c product{};
    fortai_mma_i8(product, left, right);
    const int row = warp / 2 * 32 + N * 16 + lane / 4;
    const int row_pair = row + 8;
    const int token = (warp & 1) * 8 + J0 + (lane & 3) * 2;
    const int slot = (J0 / 8 + N) * 4;
    const float dA0 = x_df[row * x_stride + group * 4 + k01 / 8];
    const float dA1 = x_df[row_pair * x_stride + group * 4 + k01 / 8];
    sum[slot + 0] += static_cast<float>(product.x[0]) * dA0 *
        y_df[token * y_stride + k01 / 8];
    sum[slot + 1] += static_cast<float>(product.x[1]) * dA0 *
        y_df[(token + 1) * y_stride + k01 / 8];
    sum[slot + 2] += static_cast<float>(product.x[2]) * dA1 *
        y_df[token * y_stride + k01 / 8];
    sum[slot + 3] += static_cast<float>(product.x[3]) * dA1 *
        y_df[(token + 1) * y_stride + k01 / 8];
}

template <int J0>
__device__ __forceinline__ void fortai_mma_accumulate_j0(
    const fortai_mma_tile_a (&left)[2], const int *tile_y, const float *x_df,
    const float *y_df, int x_stride, int y_stride, int group, int k01,
    int warp, int lane, float *sum) {
    fortai_mma_tile_b right;
    const int token_base = (warp & 1) * 8;
    fortai_mma_load_b_generic(right, tile_y + 4 + (token_base + J0) * y_stride + k01,
        y_stride);
    fortai_mma_accumulate_product<J0, 0>(left[0], right, x_df, y_df,
        x_stride, y_stride, group, k01, warp, lane, sum);
    fortai_mma_accumulate_product<J0, 1>(left[1], right, x_df, y_df,
        x_stride, y_stride, group, k01, warp, lane, sum);
}

template <int J0, int N>
__device__ __forceinline__ void fortai_mma_accumulate_product_scaled(
    const fortai_mma_tile_a &left, const fortai_mma_tile_b &right,
    float dA0, float dA1, float dB0, float dB1, float *sum) {
    fortai_mma_tile_c product{};
    fortai_mma_i8(product, left, right);
    const int slot = (J0 / 8 + N) * 4;
    sum[slot + 0] += static_cast<float>(product.x[0]) * dA0 * dB0;
    sum[slot + 1] += static_cast<float>(product.x[1]) * dA0 * dB1;
    sum[slot + 2] += static_cast<float>(product.x[2]) * dA1 * dB0;
    sum[slot + 3] += static_cast<float>(product.x[3]) * dA1 * dB1;
}

template <int J0>
__device__ __forceinline__ void fortai_mma_accumulate_j0_preloaded(
    const fortai_mma_tile_a (&left)[2][4], const float (&dA)[2][2][4],
    const int *tile_y, const float *y_df, int y_stride, int warp, int lane,
    float *sum) {
    const int token_base = (warp & 1) * 8;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int k01 = k * 8;
        fortai_mma_tile_b right;
        fortai_mma_load_b_generic(right,
            tile_y + 4 + (token_base + J0) * y_stride + k01, y_stride);
        const int token = token_base + J0 + (lane & 3) * 2;
        const float dB0 = y_df[token * y_stride + k];
        const float dB1 = y_df[(token + 1) * y_stride + k];
        fortai_mma_accumulate_product_scaled<J0, 0>(left[0][k], right,
            dA[0][0][k], dA[0][1][k], dB0, dB1, sum);
        fortai_mma_accumulate_product_scaled<J0, 1>(left[1][k], right,
            dA[1][0][k], dA[1][1][k], dB0, dB1, sum);
    }
}

template <int J0, int N, int RowsPerTile, int TokensPerTile>
__device__ __forceinline__ void fortai_mma_write_back(
    const float *sum, int warp, int lane, int row_tile, int token_tile,
    int rows, int batch, bool tile_complete, float *output, float *tmp_fixup,
    int block) {
    const int row = warp / 2 * 32 + N * 16 + lane / 4;
    const int row_pair = row + 8;
    const int token = (warp & 1) * 8 + J0 + (lane & 3) * 2;
    const int slot = (J0 / 8 + N) * 4;
    if (row_pair >= RowsPerTile || token + 1 >= TokensPerTile ||
        row_tile + row_pair >= rows || token_tile + token + 1 >= batch)
        return;
    const size_t row0 = static_cast<size_t>(row_tile + row);
    const size_t row1 = static_cast<size_t>(row_tile + row_pair);
    const size_t token0 = static_cast<size_t>(token_tile + token);
    const size_t token1 = token0 + 1;
    if (tile_complete) {
        output[token0 * rows + row0] = sum[slot + 0];
        output[token1 * rows + row0] = sum[slot + 1];
        output[token0 * rows + row1] = sum[slot + 2];
        output[token1 * rows + row1] = sum[slot + 3];
    } else {
        const size_t base = static_cast<size_t>(block) * RowsPerTile * TokensPerTile;
        tmp_fixup[base + token * RowsPerTile + row] = sum[slot + 0];
        tmp_fixup[base + (token + 1) * RowsPerTile + row] = sum[slot + 1];
        tmp_fixup[base + token * RowsPerTile + row_pair] = sum[slot + 2];
        tmp_fixup[base + (token + 1) * RowsPerTile + row_pair] = sum[slot + 3];
    }
}

/* Stream-K can split one output tile across resident blocks.  In that case
 * every partial is accumulated into the zeroed output tile.  The operation is
 * deliberately kept separate from the normal assignment/fixup writer so the
 * tiled fast path retains its race-free stores and no extra branch enters its
 * hot loop. */
template <int J0, int N, int RowsPerTile, int TokensPerTile>
__device__ __forceinline__ void fortai_mma_write_back_atomic(
    const float *sum, int warp, int lane, int row_tile, int token_tile,
    int rows, int batch, float *output) {
    const int row = warp / 2 * 32 + N * 16 + lane / 4;
    const int row_pair = row + 8;
    const int token = (warp & 1) * 8 + J0 + (lane & 3) * 2;
    const int slot = (J0 / 8 + N) * 4;
    if (row_pair >= RowsPerTile || token + 1 >= TokensPerTile ||
        row_tile + row_pair >= rows || token_tile + token + 1 >= batch)
        return;
    const size_t row0 = static_cast<size_t>(row_tile + row);
    const size_t row1 = static_cast<size_t>(row_tile + row_pair);
    const size_t token0 = static_cast<size_t>(token_tile + token);
    const size_t token1 = token0 + 1;
    atomicAdd(output + token0 * rows + row0, sum[slot + 0]);
    atomicAdd(output + token1 * rows + row0, sum[slot + 1]);
    atomicAdd(output + token0 * rows + row1, sum[slot + 2]);
    atomicAdd(output + token1 * rows + row1, sum[slot + 3]);
}

template <int J0, int N, int RowsPerTile, int TokensPerTile>
__device__ __forceinline__ void fortai_mma_write_back_stream(
    const float *sum, int warp, int lane, int row_tile, int token_tile,
    int rows, int batch, bool direct_tile, float *output, float *tmp_fixup,
    int block) {
    fortai_mma_write_back<J0, N, RowsPerTile, TokensPerTile>(sum, warp, lane,
        row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup, block);
}

template <int SumStride>
__device__ __forceinline__ void fortai_mma_accumulate_product_strided(
    const fortai_mma_tile_a &left, const fortai_mma_tile_b &right,
    int slot, float dA0, float dA1, float dB0, float dB1, float *sum) {
    fortai_mma_tile_c product{};
    fortai_mma_i8(product, left, right);
    sum[(slot + 0) * SumStride] += static_cast<float>(product.x[0]) * dA0 * dB0;
    sum[(slot + 1) * SumStride] += static_cast<float>(product.x[1]) * dA0 * dB1;
    sum[(slot + 2) * SumStride] += static_cast<float>(product.x[2]) * dA1 * dB0;
    sum[(slot + 3) * SumStride] += static_cast<float>(product.x[3]) * dA1 * dB1;
}

template <int SumStride>
__device__ __forceinline__ void fortai_mma_accumulate_j0_strided(
    const fortai_mma_tile_a (&left)[2], const int *tile_y, const float *y_df,
    int y_stride, int k01, int warp, int lane, int j0,
    float dA00, float dA01, float dA10, float dA11, float *sum) {
    fortai_mma_tile_b right;
    const int token_base = (warp & 1) * 8;
    fortai_mma_load_b_generic(right, tile_y + 4 + (token_base + j0) * y_stride + k01,
        y_stride);
    const int token = token_base + j0 + (lane & 3) * 2;
    const float dB0 = y_df[token * y_stride + k01 / 8];
    const float dB1 = y_df[(token + 1) * y_stride + k01 / 8];
    const int slot = (j0 / 8) * 4;
    fortai_mma_accumulate_product_strided<SumStride>(left[0], right, slot,
        dA00, dA01, dB0, dB1, sum);
    fortai_mma_accumulate_product_strided<SumStride>(left[1], right, slot + 4,
        dA10, dA11, dB0, dB1, sum);
}

template <int RowsPerTile, bool FullTiles>
__device__ __forceinline__ void fortai_mma_write_back_strided(
    const float *sum, int warp, int lane, int row_tile, int token_tile,
    int rows, int batch, int token_offset, float *output) {
    constexpr int tokens_per_half = 64;
    constexpr int sum_stride = 256;
    const int row_base = warp / 2 * 32 + lane / 4;
    const int token_base = (warp & 1) * 8 + (lane & 3) * 2;
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        const int row = row_base + n * 16;
        const int row_pair = row + 8;
#pragma unroll
        for (int j0 = 0; j0 < tokens_per_half / 16; ++j0) {
            const int token = token_base + j0 * 16;
            const int slot = (j0 * 2 + n) * 4;
            const bool valid = FullTiles || (row_tile + row_pair < rows &&
                token_tile + token_offset + token + 1 < batch);
            if (!valid) continue;
            output[static_cast<size_t>(token_tile + token_offset + token) * rows +
                row_tile + row] = sum[(slot + 0) * sum_stride];
            output[static_cast<size_t>(token_tile + token_offset + token + 1) * rows +
                row_tile + row] = sum[(slot + 1) * sum_stride];
            output[static_cast<size_t>(token_tile + token_offset + token) * rows +
                row_tile + row_pair] = sum[(slot + 2) * sum_stride];
            output[static_cast<size_t>(token_tile + token_offset + token + 1) * rows +
                row_tile + row_pair] = sum[(slot + 3) * sum_stride];
        }
    }
}

/* Spill-free J=128 Q8 MMA tile.  The first 64 token columns are accumulated
 * in a transposed shared slab (one coalesced bank access per lane), while the
 * second 64 columns stay in registers.  This preserves one block per 128
 * output tokens but avoids the 128-byte local frame emitted for the fully
 * register-resident wide tile on sm_120. */
template <int RowsPerTile = 128, bool FullTiles = false>
__launch_bounds__(256, 1)
__global__ void q8_gemm_batch_mmq_mma_layout_wide_split(
    const int8_t *__restrict__ weights, const int8_t *__restrict__ activation,
    int rows, int blocks, int batch, float *__restrict__ output,
    float *__restrict__ tmp_fixup, int32_t *__restrict__ tile_ids) {
    constexpr int nwarps = 8;
    constexpr int tokens_per_half = 64;
    constexpr int q8_x_stride = 2 * q8_block_width + 4 + 8;
    constexpr int q8_x_scale_offset = 2 * q8_block_width;
    constexpr int y_stride = 4 + q8_block_width;
    constexpr int threads = nwarps * 32;
    constexpr int sum_values = tokens_per_half * RowsPerTile / (nwarps * 32);
    constexpr int tile_y_offset = tokens_per_half;
    constexpr int tile_x_offset = tile_y_offset + tokens_per_half * y_stride;
    constexpr int tile_x_values = RowsPerTile * q8_x_stride;
    constexpr size_t shared_bytes = static_cast<size_t>(tile_x_offset + tile_x_values) *
        sizeof(int) + static_cast<size_t>(threads * sum_values) * sizeof(float);
    static_assert(shared_bytes == 81152, "unexpected wide split shared layout");

    extern __shared__ unsigned char storage[];
    auto *tile_y = reinterpret_cast<int *>(storage) + tile_y_offset;
    auto *tile_x = reinterpret_cast<int *>(storage) + tile_x_offset;
    auto *x_df = reinterpret_cast<float *>(tile_x + q8_x_scale_offset);
    auto *y_df = reinterpret_cast<float *>(tile_y);
    auto *sum_shared = reinterpret_cast<float *>(tile_x + tile_x_values);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int linear_tid = warp * 32 + lane;
    const int row_tiles = (rows + RowsPerTile - 1) / RowsPerTile;
    const int token_tiles = (batch + 127) / 128;
    const int tile_count = row_tiles * token_tiles;
    const int tile_index = static_cast<int>(blockIdx.x);
    if (tile_index >= tile_count) return;
    const int row_tile = (tile_index % row_tiles) * RowsPerTile;
    const int token_tile = (tile_index / row_tiles) * 128;
    (void) tmp_fixup;
    (void) tile_ids;

    float sum_register[sum_values] = {};
    float *sum_thread = sum_shared + linear_tid;
    for (int slot = 0; slot < sum_values; ++slot)
        sum_thread[slot * threads] = 0.0f;
    __syncthreads();

    for (int k_base = 0; k_base < blocks; k_base += 8) {
        for (int i0 = 0; i0 < RowsPerTile; i0 += nwarps) {
            const int row = i0 + warp;
            const int source_row = row_tile + row;
            const int block = lane / 8;
            const int word = lane & 7;
            const int source_block = k_base + block;
            const bool valid = FullTiles || (source_row < rows && source_block < blocks);
            tile_x[row * q8_x_stride + lane] = valid ?
                load_i32(weights + (static_cast<size_t>(source_row) * blocks + source_block) *
                    q8_block_bytes + 2 + word * 4) : 0;
            const int second_block = k_base + block + 4;
            const bool second_valid = FullTiles ||
                (source_row < rows && second_block < blocks);
            tile_x[row * q8_x_stride + q8_block_width + lane] = second_valid ?
                load_i32(weights + (static_cast<size_t>(source_row) * blocks + second_block) *
                    q8_block_bytes + 2 + word * 4) : 0;
        }
        for (int i0 = 0; i0 < RowsPerTile; i0 += nwarps * 4) {
            const int row = i0 + warp * 4 + lane / 8;
            const int block = lane & 7;
            const int source_block = k_base + block;
            const bool valid = FullTiles ||
                (row_tile + row < rows && source_block < blocks);
            x_df[row * q8_x_stride + block] = valid ? block_scale(weights +
                (static_cast<size_t>(row_tile + row) * blocks + source_block) *
                    q8_block_bytes) : 0.0f;
        }
        __syncthreads();

        for (int group = 0; group < 2; ++group) {
            const int activation_group = k_base / 4 + group;
            for (int half = 0; half < 2; ++half) {
                for (int index = linear_tid; index < tokens_per_half * y_stride;
                    index += threads) {
                    const int token = index / y_stride;
                    const int word = index % y_stride;
                    const int source_token = token_tile + half * tokens_per_half + token;
                    const bool valid = FullTiles || (source_token < batch &&
                        activation_group < (blocks + 3) / 4);
                    tile_y[token * y_stride + word] = valid ?
                        reinterpret_cast<const int *>(activation)[
                            (static_cast<size_t>(activation_group) * batch + source_token) *
                                y_stride + word] : 0;
                }
                __syncthreads();

#pragma unroll
                for (int k01 = 0; k01 < q8_block_width; k01 += 8) {
                    const int x_k = group * q8_block_width + k01;
                    fortai_mma_tile_a left[2];
#pragma unroll
                    for (int n = 0; n < 2; ++n)
                        fortai_mma_load_a(left[n], tile_x +
                            (warp / 2 * 32 + n * 16) * q8_x_stride + x_k, q8_x_stride);

                    const int row0 = warp / 2 * 32 + lane / 4;
                    const int row1 = row0 + 8;
                    const int row2 = row0 + 16;
                    const int row3 = row2 + 8;
                    const int scale_index = group * 4 + k01 / 8;
                    const float dA00 = x_df[row0 * q8_x_stride + scale_index];
                    const float dA01 = x_df[row1 * q8_x_stride + scale_index];
                    const float dA10 = x_df[row2 * q8_x_stride + scale_index];
                    const float dA11 = x_df[row3 * q8_x_stride + scale_index];
                    if (half == 0) {
                        fortai_mma_accumulate_j0_strided<threads>(left, tile_y, y_df,
                            y_stride, k01, warp, lane, 0, dA00, dA01, dA10, dA11, sum_thread);
                        fortai_mma_accumulate_j0_strided<threads>(left, tile_y, y_df,
                            y_stride, k01, warp, lane, 16, dA00, dA01, dA10, dA11, sum_thread);
                        fortai_mma_accumulate_j0_strided<threads>(left, tile_y, y_df,
                            y_stride, k01, warp, lane, 32, dA00, dA01, dA10, dA11, sum_thread);
                        fortai_mma_accumulate_j0_strided<threads>(left, tile_y, y_df,
                            y_stride, k01, warp, lane, 48, dA00, dA01, dA10, dA11, sum_thread);
                    } else {
                        fortai_mma_accumulate_j0<0>(left, tile_y, x_df, y_df,
                            q8_x_stride, y_stride, group, k01, warp, lane, sum_register);
                        fortai_mma_accumulate_j0<16>(left, tile_y, x_df, y_df,
                            q8_x_stride, y_stride, group, k01, warp, lane, sum_register);
                        fortai_mma_accumulate_j0<32>(left, tile_y, x_df, y_df,
                            q8_x_stride, y_stride, group, k01, warp, lane, sum_register);
                        fortai_mma_accumulate_j0<48>(left, tile_y, x_df, y_df,
                            q8_x_stride, y_stride, group, k01, warp, lane, sum_register);
                    }
                }
                __syncthreads();
            }
        }
    }

    fortai_mma_write_back_strided<RowsPerTile, FullTiles>(sum_shared + linear_tid, warp, lane,
        row_tile, token_tile, rows, batch, 0, output);
    fortai_mma_write_back<0, 0, RowsPerTile, 64>(sum_register, warp, lane,
        row_tile, token_tile + 64, rows, batch, true, output, tmp_fixup,
        static_cast<int>(blockIdx.x));
    fortai_mma_write_back<0, 1, RowsPerTile, 64>(sum_register, warp, lane,
        row_tile, token_tile + 64, rows, batch, true, output, tmp_fixup,
        static_cast<int>(blockIdx.x));
    fortai_mma_write_back<16, 0, RowsPerTile, 64>(sum_register, warp, lane,
        row_tile, token_tile + 64, rows, batch, true, output, tmp_fixup,
        static_cast<int>(blockIdx.x));
    fortai_mma_write_back<16, 1, RowsPerTile, 64>(sum_register, warp, lane,
        row_tile, token_tile + 64, rows, batch, true, output, tmp_fixup,
        static_cast<int>(blockIdx.x));
    fortai_mma_write_back<32, 0, RowsPerTile, 64>(sum_register, warp, lane,
        row_tile, token_tile + 64, rows, batch, true, output, tmp_fixup,
        static_cast<int>(blockIdx.x));
    fortai_mma_write_back<32, 1, RowsPerTile, 64>(sum_register, warp, lane,
        row_tile, token_tile + 64, rows, batch, true, output, tmp_fixup,
        static_cast<int>(blockIdx.x));
    fortai_mma_write_back<48, 0, RowsPerTile, 64>(sum_register, warp, lane,
        row_tile, token_tile + 64, rows, batch, true, output, tmp_fixup,
        static_cast<int>(blockIdx.x));
    fortai_mma_write_back<48, 1, RowsPerTile, 64>(sum_register, warp, lane,
        row_tile, token_tile + 64, rows, batch, true, output, tmp_fixup,
        static_cast<int>(blockIdx.x));
}

template <int RowsPerTile = 128, int TokensPerTile = 64, int BlocksPerTile = 8,
    bool FullTiles = false>
__global__ void q8_gemm_batch_mmq_mma_layout(const int8_t *__restrict__ weights,
    const int8_t *__restrict__ activation, int rows, int blocks, int batch,
    int activation_stride,
    float *__restrict__ output, float *__restrict__ tmp_fixup,
    int32_t *__restrict__ tile_ids) {
    constexpr int nwarps = 8;
    constexpr int x_stride = 2 * q8_block_width + 4 + 8;
    constexpr int x_scale_offset = 2 * q8_block_width;
    constexpr int y_stride = 4 + q8_block_width;
    constexpr int tile_y_offset = TokensPerTile;
    constexpr int tile_x_offset = tile_y_offset + TokensPerTile * y_stride;
    constexpr int shared_bytes = tile_x_offset * static_cast<int>(sizeof(int)) +
        RowsPerTile * x_stride * static_cast<int>(sizeof(int));
    static_assert(shared_bytes == static_cast<size_t>(TokensPerTile) *
            (4 + q8_block_width + 1) * sizeof(int) +
            static_cast<size_t>(RowsPerTile) * (2 * q8_block_width + 4 + 8) * sizeof(int),
        "unexpected native MMA shared layout");

    extern __shared__ unsigned char storage[];
    auto *tile_y = reinterpret_cast<int *>(storage) + tile_y_offset;
    auto *tile_x = reinterpret_cast<int *>(storage) + tile_x_offset;
    auto *x_df = reinterpret_cast<float *>(tile_x + x_scale_offset);
    auto *y_df = reinterpret_cast<float *>(tile_y);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int linear_tid = warp * 32 + lane;
    const int row_tiles = (rows + RowsPerTile - 1) / RowsPerTile;
    const int token_tiles = (batch + TokensPerTile - 1) / TokensPerTile;
    const int total_tiles = row_tiles * token_tiles;
    const int k_tiles = (blocks + BlocksPerTile - 1) / BlocksPerTile;
    const int total_units = total_tiles * k_tiles;
    (void) tmp_fixup;
    (void) tile_ids;

    const int unit_start = static_cast<int>(blockIdx.x) * total_units /
        static_cast<int>(gridDim.x);
    const int unit_stop = static_cast<int>(blockIdx.x + 1) * total_units /
        static_cast<int>(gridDim.x);
    if (linear_tid == 0) tile_ids[blockIdx.x] = -1;
    __syncthreads();

    int unit = unit_start;
    while (unit < unit_stop) {
        const int tile_index = unit / k_tiles;
        const int tile_unit_stop = min(unit_stop, (tile_index + 1) * k_tiles);
        const int row_tile = (tile_index % row_tiles) * RowsPerTile;
        const int token_tile = (tile_index / row_tiles) * TokensPerTile;
        /* rows_per_warp is 32, so each warp owns two 16-row MMA fragments.
         * The token-fragment count scales with J (four groups for J=64 and
         * eight for J=128), while each fragment still uses four C values. */
        constexpr int sum_elements = TokensPerTile * RowsPerTile / (nwarps * 32);
        /* A stream-K block normally finishes one output tile, but a
         * high-work launch can cover several complete tiles.  The
         * accumulator belongs to the tile, not to the CUDA block: reset it
         * once per tile, while retaining it across that tile's K slices. */
        float sum[sum_elements] = {};

        for (; unit < tile_unit_stop; ++unit) {
            const int k_base = (unit % k_tiles) * BlocksPerTile;

            for (int i0 = 0; i0 < RowsPerTile; i0 += nwarps) {
                const int row = i0 + warp;
                const int source_row = row_tile + row;
                const int block = lane / 8;
                const int word = lane & 7;
                const int source_block = k_base + block;
                const bool valid = FullTiles ||
                    (source_row < rows && source_block < blocks);
                tile_x[row * x_stride + lane] = valid ?
                    load_i32(weights + (static_cast<size_t>(source_row) * blocks +
                        source_block) * q8_block_bytes + 2 + word * 4) : 0;

                const int second_block = k_base + block + 4;
                const bool second_valid = FullTiles ||
                    (source_row < rows && second_block < blocks);
                tile_x[row * x_stride + q8_block_width + lane] = second_valid ?
                    load_i32(weights + (static_cast<size_t>(source_row) * blocks +
                        second_block) * q8_block_bytes + 2 + word * 4) : 0;
            }

            for (int i0 = 0; i0 < RowsPerTile; i0 += nwarps * 4) {
                const int row = i0 + warp * 4 + lane / 8;
                const int block = lane & 7;
                const int source_block = k_base + block;
                const bool valid = FullTiles ||
                    (row_tile + row < rows && source_block < blocks);
                x_df[row * x_stride + block] = valid ? block_scale(weights +
                    (static_cast<size_t>(row_tile + row) * blocks + source_block) *
                        q8_block_bytes) : 0.0f;
            }
            __syncthreads();

            #pragma unroll
            for (int group = 0; group < 2; ++group) {
                const int activation_group = k_base / 4 + group;
                for (int index = linear_tid; index < TokensPerTile * y_stride;
                    index += nwarps * 32) {
                    const int token = index / y_stride;
                    const int word = index % y_stride;
                    const int source_token = token_tile + token;
                    const bool valid = FullTiles || (source_token < batch &&
                        activation_group < (blocks + 3) / 4);
                    tile_y[token * y_stride + word] = valid ?
                        reinterpret_cast<const int *>(activation)[
                            (static_cast<size_t>(activation_group) * activation_stride +
                                source_token) * y_stride + word] : 0;
                }
                __syncthreads();

                fortai_mma_tile_a left[2][4];
                float dA[2][2][4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int k01 = k * 8;
                    const int x_k = group * q8_block_width + k01;
#pragma unroll
                    for (int n = 0; n < 2; ++n)
                        fortai_mma_load_a(left[n][k], tile_x +
                            (warp / 2 * 32 + n * 16) * x_stride + x_k,
                            x_stride);
                    const int row0 = warp / 2 * 32 + lane / 4;
                    const int row1 = row0 + 8;
                    const int row2 = row0 + 16;
                    const int row3 = row2 + 8;
                    const int scale_index = group * 4 + k;
                    dA[0][0][k] = x_df[row0 * x_stride + scale_index];
                    dA[0][1][k] = x_df[row1 * x_stride + scale_index];
                    dA[1][0][k] = x_df[row2 * x_stride + scale_index];
                    dA[1][1][k] = x_df[row3 * x_stride + scale_index];
                }
                fortai_mma_accumulate_j0_preloaded<0>(left, dA, tile_y, y_df,
                    y_stride, warp, lane, sum);
                fortai_mma_accumulate_j0_preloaded<16>(left, dA, tile_y, y_df,
                    y_stride, warp, lane, sum);
                fortai_mma_accumulate_j0_preloaded<32>(left, dA, tile_y, y_df,
                    y_stride, warp, lane, sum);
                fortai_mma_accumulate_j0_preloaded<48>(left, dA, tile_y, y_df,
                    y_stride, warp, lane, sum);
                if constexpr (TokensPerTile >= 128) {
                    fortai_mma_accumulate_j0_preloaded<64>(left, dA, tile_y, y_df,
                        y_stride, warp, lane, sum);
                    fortai_mma_accumulate_j0_preloaded<80>(left, dA, tile_y, y_df,
                        y_stride, warp, lane, sum);
                    fortai_mma_accumulate_j0_preloaded<96>(left, dA, tile_y, y_df,
                        y_stride, warp, lane, sum);
                    fortai_mma_accumulate_j0_preloaded<112>(left, dA, tile_y, y_df,
                        y_stride, warp, lane, sum);
                }
                __syncthreads();
            }
        }
        /* In stream-K, only the final chunk of a block that stops in the
         * middle of an output tile goes to the per-block fixup buffer.  All
         * other chunks can store directly, exactly as llama.cpp's partition
         * does; this avoids atomics while preserving split-tile correctness. */
        const bool fixup_partial = tile_unit_stop == unit_stop &&
            tile_unit_stop < (tile_index + 1) * k_tiles;
        const bool direct_tile = !fixup_partial;
        fortai_mma_write_back_stream<0, 0, RowsPerTile, TokensPerTile>(sum, warp, lane,
            row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
            static_cast<int>(blockIdx.x));
        fortai_mma_write_back_stream<0, 1, RowsPerTile, TokensPerTile>(sum, warp, lane,
            row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
            static_cast<int>(blockIdx.x));
        fortai_mma_write_back_stream<16, 0, RowsPerTile, TokensPerTile>(sum, warp, lane,
            row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
            static_cast<int>(blockIdx.x));
        fortai_mma_write_back_stream<16, 1, RowsPerTile, TokensPerTile>(sum, warp, lane,
            row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
            static_cast<int>(blockIdx.x));
        fortai_mma_write_back_stream<32, 0, RowsPerTile, TokensPerTile>(sum, warp, lane,
            row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
            static_cast<int>(blockIdx.x));
        fortai_mma_write_back_stream<32, 1, RowsPerTile, TokensPerTile>(sum, warp, lane,
            row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
            static_cast<int>(blockIdx.x));
        fortai_mma_write_back_stream<48, 0, RowsPerTile, TokensPerTile>(sum, warp, lane,
            row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
            static_cast<int>(blockIdx.x));
        fortai_mma_write_back_stream<48, 1, RowsPerTile, TokensPerTile>(sum, warp, lane,
            row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
            static_cast<int>(blockIdx.x));
        if constexpr (TokensPerTile >= 128) {
            fortai_mma_write_back_stream<64, 0, RowsPerTile, TokensPerTile>(sum, warp, lane,
                row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
                static_cast<int>(blockIdx.x));
            fortai_mma_write_back_stream<64, 1, RowsPerTile, TokensPerTile>(sum, warp, lane,
                row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
                static_cast<int>(blockIdx.x));
            fortai_mma_write_back_stream<80, 0, RowsPerTile, TokensPerTile>(sum, warp, lane,
                row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
                static_cast<int>(blockIdx.x));
            fortai_mma_write_back_stream<80, 1, RowsPerTile, TokensPerTile>(sum, warp, lane,
                row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
                static_cast<int>(blockIdx.x));
            fortai_mma_write_back_stream<96, 0, RowsPerTile, TokensPerTile>(sum, warp, lane,
                row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
                static_cast<int>(blockIdx.x));
            fortai_mma_write_back_stream<96, 1, RowsPerTile, TokensPerTile>(sum, warp, lane,
                row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
                static_cast<int>(blockIdx.x));
            fortai_mma_write_back_stream<112, 0, RowsPerTile, TokensPerTile>(sum, warp, lane,
                row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
                static_cast<int>(blockIdx.x));
            fortai_mma_write_back_stream<112, 1, RowsPerTile, TokensPerTile>(sum, warp, lane,
                row_tile, token_tile, rows, batch, direct_tile, output, tmp_fixup,
                static_cast<int>(blockIdx.x));
        }
        if (fixup_partial && linear_tid == 0) tile_ids[blockIdx.x] = tile_index;
        __syncthreads();
    }
}

template <int RowsPerTile = 128, int TokensPerTile = 64>
__global__ void q8_gemm_batch_mmq_fixup(const float *__restrict__ tmp_fixup,
    const int32_t *__restrict__ tile_ids, int rows, int batch, int row_tiles,
    int tile_count, int grid_blocks, float *__restrict__ output) {
    const int tile_index = static_cast<int>(blockIdx.y);
    if (tile_index >= tile_count) return;
    const int row_tile = (tile_index % row_tiles) * RowsPerTile;
    const int token_tile = (tile_index / row_tiles) * TokensPerTile;
    const int first_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int index = first_index; index < RowsPerTile * TokensPerTile; index += stride) {
        const int row = index % RowsPerTile;
        const int token = index / RowsPerTile;
        if (row_tile + row >= rows || token_tile + token >= batch) continue;
        float sum = 0.0f;
        for (int block = 0; block < grid_blocks; ++block) {
            if (tile_ids[block] == tile_index)
                sum += tmp_fixup[static_cast<size_t>(block) * RowsPerTile * TokensPerTile + index];
        }
        if (sum != 0.0f)
            output[static_cast<size_t>(token_tile + token) * rows + row_tile + row] += sum;
    }
}

__global__ void q8_repack_activation(const int8_t *raw, int8_t *padded, int blocks) {
    const int block = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (block >= blocks || lane >= q8_block_width) return;
    const int raw_offset = block * q8_block_bytes;
    const int padded_offset = block * q8_activation_block_bytes;
    if (lane < 2) padded[padded_offset + lane] = raw[raw_offset + lane];
    padded[padded_offset + q8_activation_data_offset + lane] = raw[raw_offset + 2 + lane];
}

static void launch_q8_repack_activation(const int8_t *raw, int8_t *padded,
    int blocks, cudaStream_t stream) {
    q8_repack_activation<<<blocks, 32, 0, stream>>>(raw, padded, blocks);
}

__global__ void qwen_embedding_lookup_q8(const int8_t *weights, int64_t token_id,
    int blocks, float *output) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int elements = blocks * q8_block_width;
    if (index >= elements) return;
    const int block = index / q8_block_width;
    const int offset = index % q8_block_width;
    const int8_t *weight_block = weights + static_cast<size_t>(token_id) * blocks * q8_block_bytes +
        block * q8_block_bytes;
    output[index] = block_scale(weight_block) * static_cast<float>(weight_block[2 + offset]);
}

__global__ void qwen_embedding_lookup_q8_batch(const int8_t *weights,
    const int32_t *tokens, int blocks, int batch, float *output) {
    const int token = static_cast<int>(blockIdx.y);
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int elements = blocks * q8_block_width;
    if (token >= batch || index >= elements) return;
    const int block = index / q8_block_width;
    const int offset = index % q8_block_width;
    const int64_t token_id = static_cast<int64_t>(tokens[token]);
    const int8_t *weight_block = weights +
        static_cast<size_t>(token_id) * blocks * q8_block_bytes + block * q8_block_bytes;
    output[static_cast<size_t>(token) * elements + index] =
        block_scale(weight_block) * static_cast<float>(weight_block[2 + offset]);
}

__global__ void qwen_add_float(const float *left, const float *right, float *output,
    int elements) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) output[index] = left[index] + right[index];
}

__global__ void qwen_copy_column(const float *input, int stride, int column,
    float *output, int elements) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements)
        output[index] = input[static_cast<size_t>(column) * stride + index];
}

__global__ void qwen_add_matrix(const float *left, const float *right,
    float *output, int elements) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) output[index] = left[index] + right[index];
}

__global__ void qwen_silu_product_matrix(float *gate, const float *up,
    int elements) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) gate[index] = qwen_silu(gate[index]) * up[index];
}

__global__ void qwen_rms_norm_float(const float *__restrict__ input,
    const float *__restrict__ weights, float *__restrict__ output,
    int elements, float epsilon) {
    /* Keep the oracle's double accumulator and F32 square, but parallelize
     * the load/reduction.  The resulting error is below one F32 ulp while
     * avoiding a 40-KiB per-vector square staging buffer. */
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ double partial[32];
    __shared__ float inverse_scale;
    double sum_squares = 0.0;
    for (int index = tid; index < elements; index += blockDim.x) {
        const float value = input[index];
        sum_squares += static_cast<double>(value * value);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_squares += __shfl_down_sync(0xffffffffu, sum_squares, offset);
    if (lane == 0) partial[warp] = sum_squares;
    __syncthreads();
    if (tid == 0) {
        sum_squares = 0.0;
        const int warps = (blockDim.x + 31) / 32;
        for (int index = 0; index < warps; ++index)
            sum_squares += partial[index];
        const float mean = static_cast<float>(sum_squares / static_cast<double>(elements));
        inverse_scale = 1.0f / sqrtf(mean + epsilon);
    }
    __syncthreads();
    for (int index = tid; index < elements; index += blockDim.x)
    {
        const volatile float scaled = input[index] * inverse_scale;
        output[index] = scaled * weights[index];
    }
}

/* The CUDA path is already the production numerical implementation; keep its
 * reduction in FP32, as ggml-cuda's rms_norm_f32 kernel does.  The reference
 * CPU path intentionally retains the wider accumulator for oracle checks, but
 * using double shuffle/reduction values here needlessly cuts occupancy on
 * consumer SMs.  The final inverse scale is still formed in FP32, so this
 * path has the same model-level rounding boundary as llama.cpp. */
__global__ void qwen_rms_norm_float_fast(const float *__restrict__ input,
    const float *__restrict__ weights, float *__restrict__ output,
    int elements, float epsilon) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float partial[32];
    __shared__ float inverse_scale;
    float sum_squares = 0.0f;
    for (int index = tid; index < elements; index += blockDim.x) {
        const float value = input[index];
        sum_squares = fmaf(value, value, sum_squares);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_squares += __shfl_down_sync(0xffffffffu, sum_squares, offset);
    if (lane == 0) partial[warp] = sum_squares;
    __syncthreads();
    if (tid == 0) {
        sum_squares = 0.0f;
        const int warps = (blockDim.x + 31) / 32;
        for (int index = 0; index < warps; ++index)
            sum_squares += partial[index];
        inverse_scale = rsqrtf(sum_squares / static_cast<float>(elements) + epsilon);
    }
    __syncthreads();
    for (int index = tid; index < elements; index += blockDim.x)
        output[index] = input[index] * inverse_scale * weights[index];
}

__global__ void qwen_rms_norm_matrix(const float *__restrict__ input,
    const float *__restrict__ weights, float *__restrict__ output,
    int hidden, int batch, float epsilon) {
    const int token = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (token >= batch) return;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ double partial[32];
    __shared__ float inverse_scale;
    const size_t base = static_cast<size_t>(token) * hidden;
    double sum_squares = 0.0;
    for (int index = tid; index < hidden; index += blockDim.x) {
        const float value = input[base + index];
        sum_squares += static_cast<double>(value * value);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_squares += __shfl_down_sync(0xffffffffu, sum_squares, offset);
    if (lane == 0) partial[warp] = sum_squares;
    __syncthreads();
    if (tid == 0) {
        sum_squares = 0.0;
        const int warps = (blockDim.x + 31) / 32;
        for (int index = 0; index < warps; ++index)
            sum_squares += partial[index];
        const float mean = static_cast<float>(sum_squares / static_cast<double>(hidden));
        inverse_scale = 1.0f / sqrtf(mean + epsilon);
    }
    __syncthreads();
    for (int index = tid; index < hidden; index += blockDim.x)
    {
        const volatile float scaled = input[base + index] * inverse_scale;
        output[base + index] = scaled * weights[index];
    }
}

struct qwen_argmax_pair {
    float value;
    int index;
};

constexpr int qwen_topk_max = 32;

__device__ __forceinline__ uint64_t qwen_topk_key(float value, int index) {
    if (isnan(value)) return 0;
    const uint32_t bits = __float_as_uint(value);
    const uint32_t ordered = bits ^ ((static_cast<int32_t>(bits) >> 31) |
        UINT32_C(0x80000000));
    return (static_cast<uint64_t>(ordered) << 32) |
        static_cast<uint32_t>(INT_MAX - index);
}

template <int ItemsPerThread>
__global__ void qwen_topk_block_partials(const float *__restrict__ input,
    int row_elements, uint64_t *__restrict__ partial_keys,
    int *__restrict__ partial_indices) {
    constexpr int Threads = 256;
    constexpr int TileItems = Threads * ItemsPerThread;
    using sort_t = cub::BlockRadixSort<uint64_t, Threads, ItemsPerThread, int>;
    __shared__ typename sort_t::TempStorage storage;
    uint64_t keys[ItemsPerThread];
    int indices[ItemsPerThread];
    const int row = static_cast<int>(blockIdx.y);
    const int tile_start = static_cast<int>(blockIdx.x) * TileItems;
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int index = tile_start + static_cast<int>(threadIdx.x) + item * Threads;
        const bool valid = index < row_elements;
        indices[item] = valid ? index : INT_MAX;
        keys[item] = valid ? qwen_topk_key(
            input[static_cast<size_t>(row) * row_elements + index], index) : 0;
    }
    sort_t(storage).SortDescending(keys, indices);
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * ItemsPerThread + item;
        if (rank < qwen_topk_max) {
            const size_t offset =
                (static_cast<size_t>(row) * gridDim.x + blockIdx.x) * qwen_topk_max + rank;
            partial_keys[offset] = keys[item];
            partial_indices[offset] = indices[item];
        }
    }
}

template <int ItemsPerThread>
__global__ void qwen_topk_block_finalize(const float *__restrict__ input,
    int row_elements, const uint64_t *__restrict__ partial_keys,
    const int *__restrict__ partial_indices, int partials_per_row, int top_k,
    float *__restrict__ output_values, int *__restrict__ output_indices) {
    constexpr int Threads = 256;
    constexpr int TileItems = Threads * ItemsPerThread;
    using sort_t = cub::BlockRadixSort<uint64_t, Threads, ItemsPerThread, int>;
    __shared__ typename sort_t::TempStorage storage;
    uint64_t keys[ItemsPerThread];
    int indices[ItemsPerThread];
    const int row = static_cast<int>(blockIdx.x);
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int source = static_cast<int>(threadIdx.x) + item * Threads;
        const bool valid = source < partials_per_row && source < TileItems;
        const size_t offset = static_cast<size_t>(row) * partials_per_row + source;
        keys[item] = valid ? partial_keys[offset] : 0;
        indices[item] = valid ? partial_indices[offset] : INT_MAX;
    }
    sort_t(storage).SortDescending(keys, indices);
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * ItemsPerThread + item;
        if (rank < top_k) {
            const int index = indices[item];
            const size_t output = static_cast<size_t>(row) * top_k + rank;
            output_indices[output] = index;
            output_values[output] = input[static_cast<size_t>(row) * row_elements + index];
        }
    }
}

static int qwen_topk_radix_rows(fortai_cuda_q8_context_impl *context,
    const float *device_logits, int row_elements, int rows, int top_k,
    int *host_indices, float *host_values) {
    constexpr int items_per_thread = 8;
    constexpr int tile_items = 256 * items_per_thread;
    const int blocks_per_row = (row_elements + tile_items - 1) / tile_items;
    const int partials_per_row = blocks_per_row * qwen_topk_max;
    if (partials_per_row > 4096) return -1;

    const size_t partial_count = static_cast<size_t>(rows) * partials_per_row;
    const size_t selected_count = static_cast<size_t>(rows) * top_k;
    const size_t keys_bytes = partial_count * sizeof(uint64_t);
    const size_t partial_indices_bytes = partial_count * sizeof(int);
    const size_t values_bytes = selected_count * sizeof(float);
    const size_t indices_bytes = selected_count * sizeof(int);
    const size_t scratch_bytes = keys_bytes + partial_indices_bytes +
        values_bytes + indices_bytes;
    cudaError_t error = ensure_aux_scratch(context, scratch_bytes);
    if (error != cudaSuccess)
        return fail(context, FORTAI_CUDA_RUNTIME_ERROR, "radix top-k scratch", error);

    auto *partial_keys = reinterpret_cast<uint64_t *>(context->scratch_aux);
    auto *partial_indices = reinterpret_cast<int *>(
        reinterpret_cast<unsigned char *>(partial_keys) + keys_bytes);
    auto *device_values = reinterpret_cast<float *>(
        reinterpret_cast<unsigned char *>(partial_indices) + partial_indices_bytes);
    auto *device_indices = reinterpret_cast<int *>(
        reinterpret_cast<unsigned char *>(device_values) + values_bytes);
    error = ensure_download_host(context, values_bytes + indices_bytes);
    if (error != cudaSuccess)
        return fail(context, FORTAI_CUDA_RUNTIME_ERROR, "radix top-k download", error);
    auto *selected_values = static_cast<float *>(context->download_host);
    auto *selected_indices = reinterpret_cast<int *>(
        reinterpret_cast<unsigned char *>(selected_values) + values_bytes);

    qwen_topk_block_partials<items_per_thread><<<
        dim3(static_cast<unsigned>(blocks_per_row), static_cast<unsigned>(rows), 1),
        256, 0, context->stream>>>(device_logits, row_elements,
        partial_keys, partial_indices);
    qwen_topk_block_finalize<16><<<static_cast<unsigned>(rows), 256, 0,
        context->stream>>>(device_logits, row_elements, partial_keys,
        partial_indices, partials_per_row, top_k, device_values, device_indices);
    error = cudaGetLastError();
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(selected_values, device_values, values_bytes,
            cudaMemcpyDeviceToHost, context->stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(selected_indices, device_indices, indices_bytes,
            cudaMemcpyDeviceToHost, context->stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->stream);
    if (error != cudaSuccess)
        return fail(context, FORTAI_CUDA_RUNTIME_ERROR, "radix top-k", error);
    std::memcpy(host_values, selected_values, values_bytes);
    std::memcpy(host_indices, selected_indices, indices_bytes);
    return FORTAI_CUDA_OK;
}

__global__ void qwen_argmax_partials(const float *__restrict__ input, int elements,
    qwen_argmax_pair *__restrict__ partials) {
    extern __shared__ qwen_argmax_pair argmax_shared[];
    const int lane = static_cast<int>(threadIdx.x);
    qwen_argmax_pair best = {-FLT_MAX, INT_MAX};
    for (int index = static_cast<int>(blockIdx.x) * blockDim.x + lane;
         index < elements; index += static_cast<int>(gridDim.x) * blockDim.x) {
        const float value = input[index];
        if (value > best.value || (value == best.value && index < best.index))
            best = {value, index};
    }
    argmax_shared[lane] = best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const qwen_argmax_pair other = argmax_shared[lane + stride];
            if (other.value > argmax_shared[lane].value ||
                (other.value == argmax_shared[lane].value && other.index < argmax_shared[lane].index))
                argmax_shared[lane] = other;
        }
        __syncthreads();
    }
    if (lane == 0) partials[blockIdx.x] = argmax_shared[0];
}

__global__ void qwen_argmax_rows_partials(const float *__restrict__ input,
    int row_elements, qwen_argmax_pair *__restrict__ partials) {
    extern __shared__ qwen_argmax_pair argmax_shared[];
    const int lane = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    qwen_argmax_pair best = {-FLT_MAX, INT_MAX};
    const float *row_input = input + static_cast<size_t>(row) * row_elements;
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        index < row_elements; index += static_cast<int>(blockDim.x * gridDim.x)) {
        const float value = row_input[index];
        if (value > best.value || (value == best.value && index < best.index))
            best = {value, index};
    }
    argmax_shared[lane] = best;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const qwen_argmax_pair other = argmax_shared[lane + stride];
            if (other.value > argmax_shared[lane].value ||
                (other.value == argmax_shared[lane].value &&
                    other.index < argmax_shared[lane].index))
                argmax_shared[lane] = other;
        }
        __syncthreads();
    }
    if (lane == 0)
        partials[static_cast<size_t>(row) * gridDim.x + blockIdx.x] = argmax_shared[0];
}

__global__ void qwen_concat_float(const float *first, const float *second,
    float *output, size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = first[index];
    output[elements + index] = second[index];
}

__global__ void qwen_concat_float_matrix(const float *first, const float *second,
    float *output, int hidden, int batch) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= hidden * batch) return;
    const int row = index / hidden;
    const int column = index - row * hidden;
    const size_t output_base = static_cast<size_t>(row) * 2 * hidden;
    output[output_base + column] = first[index];
    output[output_base + hidden + column] = second[index];
}

__global__ void qwen_shift_target_hidden(const float *input, float *pending,
    float *output, int hidden, int batch) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= hidden * batch) return;
    const int row = index % hidden;
    const int column = index / hidden;
    if (column == 0) {
        const float previous = pending[row];
        pending[row] = input[static_cast<size_t>(batch - 1) * hidden + row];
        output[row] = previous;
    } else {
        output[index] = input[index - hidden];
    }
}

__device__ __forceinline__ void qwen_rope_pair(float *values, int dimension,
    int index, int position, float base) {
    const float angle = static_cast<float>(position) /
        powf(base, static_cast<float>(2 * index) / static_cast<float>(dimension));
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float first = values[index];
    const float second = values[index + dimension / 2];
    values[index] = first * cosine - second * sine;
    values[index + dimension / 2] = first * sine + second * cosine;
}

__global__ void qwen_attention_prepare(float *query, float *key, const float *value,
    const float *query_norm, const float *key_norm, __half *key_cache,
    __half *value_cache, int8_t *key_cache_q8, int8_t *value_cache_q8,
    int cache_key_type, int cache_value_type, int heads, int key_value_heads,
    int head_size, int value_size, int max_context, const int *position_ptr,
    int rope_dimension, float rope_base, float epsilon) {
    const int head = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (head >= heads) return;
    const int position = *position_ptr;
    /* Match ggml-cuda RMS normalization: float partials and a float
     * reciprocal-square root.  The former double reduction was both slower
     * and numerically different from the reference CUDA graph. */
    extern __shared__ float attention_partial_decode[];
    const int query_offset = head * 2 * head_size;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = (blockDim.x + 31) / 32;
    float query_sum = 0.0f;
    for (int i = tid; i < head_size; i += blockDim.x) {
        const float x = query[query_offset + i];
        query_sum += x * x;
    }
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const bool writes_kv = (head % group) == 0;
    float key_sum = 0.0f;
    const int key_offset = kv_head * head_size;
    if (writes_kv) {
        for (int i = tid; i < head_size; i += blockDim.x) {
            const float x = key[key_offset + i];
            key_sum += x * x;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        query_sum += __shfl_xor_sync(0xffffffffu, query_sum, offset);
        key_sum += __shfl_xor_sync(0xffffffffu, key_sum, offset);
    }
    if (lane == 0) {
        attention_partial_decode[warp] = query_sum;
        attention_partial_decode[blockDim.x + warp] = key_sum;
    }
    __syncthreads();
    if (warp == 0) {
        query_sum = lane < warps ? attention_partial_decode[lane] : 0.0f;
        key_sum = lane < warps ? attention_partial_decode[blockDim.x + lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            query_sum += __shfl_xor_sync(0xffffffffu, query_sum, offset);
            key_sum += __shfl_xor_sync(0xffffffffu, key_sum, offset);
        }
        if (lane == 0) {
            attention_partial_decode[0] = query_sum;
            attention_partial_decode[blockDim.x] = key_sum;
        }
    }
    __syncthreads();
    const float query_inverse = rsqrtf(attention_partial_decode[0] /
        static_cast<float>(head_size) + epsilon);
    for (int i = tid; i < head_size; i += blockDim.x)
        query[query_offset + i] *= query_inverse * query_norm[i];
    __syncthreads();
    if (rope_dimension > 0) {
        for (int i = tid; i < rope_dimension / 2; i += blockDim.x)
            qwen_rope_pair(query + query_offset, rope_dimension, i, position, rope_base);
    }
    if (writes_kv) {
        const float key_inverse = rsqrtf(attention_partial_decode[blockDim.x] /
            static_cast<float>(head_size) + epsilon);
        for (int i = tid; i < head_size; i += blockDim.x)
            key[key_offset + i] *= key_inverse * key_norm[i];
        __syncthreads();
        if (rope_dimension > 0) {
            for (int i = tid; i < rope_dimension / 2; i += blockDim.x)
                qwen_rope_pair(key + key_offset, rope_dimension, i, position, rope_base);
        }
        __syncthreads();
        if (cache_key_type == 1) {
            const int blocks = head_size / q8_block_width;
            const size_t row_offset = (static_cast<size_t>(position) * key_value_heads + kv_head) *
                static_cast<size_t>(blocks) * q8_block_bytes;
            for (int block = tid; block < blocks; block += blockDim.x)
                qwen_quantize_cache_block(key + key_offset,
                    key_cache_q8 + row_offset, block);
        } else if (cache_key_type == 2) {
            const int blocks = head_size / q4_block_width;
            const size_t row_offset = (static_cast<size_t>(position) * key_value_heads + kv_head) *
                static_cast<size_t>(blocks) * q4_block_bytes;
            for (int block = tid; block < blocks; block += blockDim.x)
                qwen_quantize_cache_block_q4(key + key_offset,
                    key_cache_q8 + row_offset, block);
        } else {
            for (int i = tid; i < head_size; i += blockDim.x)
                key_cache[position * key_value_heads * head_size + key_offset + i] =
                    __float2half_rn(key[key_offset + i]);
        }
        if (cache_value_type == 1) {
            const int blocks = value_size / q8_block_width;
            const size_t row_offset = (static_cast<size_t>(position) * key_value_heads + kv_head) *
                static_cast<size_t>(blocks) * q8_block_bytes;
            for (int block = tid; block < blocks; block += blockDim.x)
                qwen_quantize_cache_block(value + kv_head * value_size,
                    value_cache_q8 + row_offset, block);
        } else if (cache_value_type == 2) {
            const int blocks = value_size / q4_block_width;
            const size_t row_offset = (static_cast<size_t>(position) * key_value_heads + kv_head) *
                static_cast<size_t>(blocks) * q4_block_bytes;
            for (int block = tid; block < blocks; block += blockDim.x)
                qwen_quantize_cache_block_q4(value + kv_head * value_size,
                    value_cache_q8 + row_offset, block);
        } else {
            for (int i = tid; i < value_size; i += blockDim.x)
                value_cache[position * key_value_heads * value_size + kv_head * value_size + i] =
                    __float2half_rn(value[kv_head * value_size + i]);
        }
    }
}

/* Native specialization of llama.cpp's 128-thread vector flash-attention
 * kernel for the production Qwen3.5 D=DV=256 shape.  Eight lanes cooperate on
 * one QK dot (half2 products, XOR reduction width 8); each warp owns 32 keys
 * and four warps cover one 128-key tile.  The four key-group partials are kept
 * separate until the final reduction, matching flash_attn_ext_vec's VKQ
 * accumulation order while avoiding the old all-lanes/all-keys duplication. */
__launch_bounds__(128, 1)
__global__ void qwen_attention_apply_f16_vector(const float *__restrict__ query,
    const __half *__restrict__ key_cache, const __half *__restrict__ value_cache,
    float *__restrict__ attention, int heads, int key_value_heads, int head_size,
    int value_size, int max_context, const int *position_ptr) {
    constexpr int warp_size = 32;
    constexpr int warps = 4;
    constexpr int nthreads_kq = 8;
    constexpr int q_half2_per_lane = 16;
    constexpr float kq_max_offset = 3.0f * 0.6931f;

    const int head = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int tid = warp * warp_size + lane;
    if (head >= heads || head_size != 256 || value_size != 256) return;

    const int position = *position_ptr;
    if (position < 0 || position >= max_context) return;
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const int query_offset = head * 2 * head_size;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_size));

    /* On NVIDIA llama.cpp's F16-K vector specialization keeps Q and the
     * value partials in float2 (the half2 dot2 path is AMD-only).  Preserve
     * that exact conversion and accumulation boundary for oracle parity. */
    float2 qreg[q_half2_per_lane];
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        const int base = chunk * 32 + (lane % nthreads_kq) * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 q = reinterpret_cast<const float2 *>(query + query_offset)[base + j];
            qreg[chunk * 4 + j] = make_float2(q.x * scale, q.y * scale);
        }
    }

    /* Keep the value accumulator in FP32.  llama.cpp's vector kernel uses a
     * packed half2 accumulator on some CUDA builds, but its reduction order
     * is not bitwise stable across architectures.  The FP32 form preserves
     * the established llama.cpp token oracle on this SM while retaining the
     * same eight-lane ownership and four-warp tiling. */
    float2 vkq[q_half2_per_lane];
#pragma unroll
    for (int i = 0; i < q_half2_per_lane; ++i)
        vkq[i] = make_float2(0.0f, 0.0f);
    float kq_max = -FLT_MAX / 2.0f;
    float kq_sum = 0.0f;
    __shared__ float score_shared[warps * warp_size];

    for (int tile_start = 0; tile_start <= position; tile_start += warps * warp_size) {
        const int group = lane / nthreads_kq;
        const int sublane = lane % nthreads_kq;
        const unsigned subgroup_mask = 0xffu << (group * nthreads_kq);
        /* Each eight-lane group owns eight keys.  All eight lanes must
         * evaluate a given key so the XOR reduction sums disjoint Q/K
         * components; the previous prototype accidentally assigned a
         * different key to every lane and mixed those dots. */
#pragma unroll
        for (int local_key = 0; local_key < warp_size; ++local_key) {
            if (local_key / nthreads_kq != group) continue;
            const int key_index = tile_start + warp * warp_size + local_key;
            const bool valid = key_index <= position && key_index < max_context;
            float score = 0.0f;
            if (valid) {
                const size_t key_base = (static_cast<size_t>(key_index) * key_value_heads + kv_head) * head_size;
                const half2 *key_f16 = reinterpret_cast<const half2 *>(key_cache + key_base);
#pragma unroll
                for (int chunk = 0; chunk < 4; ++chunk) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 key_pair = __half22float2(
                            key_f16[chunk * 32 + sublane * 4 + j]);
                        const float2 query_pair = qreg[chunk * 4 + j];
                        /* The tensor-core path and the exactness-preserving
                         * scalar kernel both accumulate float products.  Do
                         * that here as well; only the key/vector ownership is
                         * changed by this specialization. */
                        score += query_pair.x * key_pair.x;
                        score += query_pair.y * key_pair.y;
                    }
                }
            }
            for (int offset = nthreads_kq / 2; offset > 0; offset >>= 1)
                score += __shfl_xor_sync(subgroup_mask, score, offset, nthreads_kq);
            if (sublane == local_key % nthreads_kq)
                score_shared[warp * warp_size + local_key] = valid ? score : -FLT_MAX / 2.0f;
        }
        __syncwarp();

        float tile_max = score_shared[warp * warp_size + lane] + kq_max_offset;
        for (int offset = warp_size / 2; offset > 0; offset >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset, warp_size));
        const float new_max = fmaxf(kq_max, tile_max);
        const float rescale = expf(kq_max - new_max);
        const int own_key = tile_start + warp * warp_size + lane;
        const bool own_valid = own_key <= position && own_key < max_context;
        const float weight = own_valid ?
            expf(score_shared[warp * warp_size + lane] - new_max) : 0.0f;
        kq_sum = kq_sum * rescale + weight;
#pragma unroll
        for (int i = 0; i < q_half2_per_lane; ++i) {
            vkq[i].x *= rescale;
            vkq[i].y *= rescale;
        }

        /* VKQ's local layout is four 32-half2 chunks.  The source vector
         * kernel visits keys in k0-major order, then the value chunks; retain
         * that order here. */
#pragma unroll
        for (int k0 = 0; k0 < warp_size; k0 += 4) {
            const int source_key = tile_start + warp * warp_size + k0 + lane / nthreads_kq;
            const float source_score = score_shared[warp * warp_size + k0 + lane / nthreads_kq];
            const float source_weight = (source_key <= position && source_key < max_context) ?
                expf(source_score - new_max) : 0.0f;
            if (source_key <= position && source_key < max_context) {
#pragma unroll
                for (int chunk = 0; chunk < 4; ++chunk) {
                    const int value_base = chunk * 32 + (lane % nthreads_kq) * 4;
                    const size_t value_offset = (static_cast<size_t>(source_key) * key_value_heads + kv_head) *
                        static_cast<size_t>(value_size);
                    const half2 *value_f16 = reinterpret_cast<const half2 *>(value_cache + value_offset);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 value = __half22float2(value_f16[value_base + j]);
                        vkq[chunk * 4 + j].x += value.x * source_weight;
                        vkq[chunk * 4 + j].y += value.y * source_weight;
                    }
                }
            }
        }
        kq_max = new_max;
    }
    __shared__ float warp_max[warps];
    __shared__ float warp_sum[warps];
    __shared__ float partial[warps * 4 * 256];
    __shared__ float global_max;
    __shared__ float global_sum;
    float warp_denominator = kq_sum;
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        warp_denominator += __shfl_xor_sync(0xffffffffu, warp_denominator, offset, warp_size);
    if (lane == 0) {
        warp_max[warp] = kq_max;
        warp_sum[warp] = warp_denominator;
    }
    __syncthreads();
    if (tid == 0) {
        float maximum = warp_max[0];
        for (int w = 1; w < warps; ++w) maximum = fmaxf(maximum, warp_max[w]);
        global_max = maximum;
        float denominator = 0.0f;
        for (int w = 0; w < warps; ++w)
            denominator += warp_sum[w] * expf(warp_max[w] - maximum);
        global_sum = denominator;
    }
    __syncthreads();

    const float warp_scale = expf(kq_max - global_max);
    const half2 warp_scale_h2 = __half2half2(__float2half_rn(warp_scale));
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int global_half2 = chunk * 32 + (lane % nthreads_kq) * 4 + j;
            const int component = 2 * global_half2;
            if (component + 1 < value_size) {
                const float2 value = make_float2(vkq[chunk * 4 + j].x * warp_scale,
                    vkq[chunk * 4 + j].y * warp_scale);
                partial[(warp * 4 + lane / nthreads_kq) * value_size + component] = value.x;
                partial[(warp * 4 + lane / nthreads_kq) * value_size + component + 1] = value.y;
            }
        }
    }
    __syncthreads();

    if (tid < 128) {
        const float *gate = query + query_offset + head_size;
#pragma unroll
        for (int component = tid; component < value_size; component += 128) {
            float total = 0.0f;
#pragma unroll
            for (int w = 0; w < warps; ++w) {
#pragma unroll
                for (int v = 0; v < 4; ++v)
                    total += partial[(w * 4 + v) * value_size + component];
            }
            attention[head * value_size + component] =
                (global_sum > 0.0f ? total / global_sum : 0.0f) * qwen_sigmoid(gate[component]);
        }
    }
}

/* Q8_0 decode specialization corresponding to llama.cpp's
 * flash_attn_ext_vec<D, 1, Q8_0, Q8_0>.  Each warp owns 32 keys from a
 * 128-key tile.  Four lanes cover one 32-value Q8 block, and each lane uses
 * two dp4a instructions for the two Q8_0 blocks selected by the Q8_1 query
 * quantizer.  The value partials use the same lane/key ownership, so the
 * whole attention operation needs only one block-wide barrier per tile. */
__launch_bounds__(128, 1)
__global__ void qwen_attention_apply_q8_vector(const float *__restrict__ query,
    const int8_t *__restrict__ key_cache, const int8_t *__restrict__ value_cache,
    float *__restrict__ attention, int heads, int key_value_heads, int head_size,
    int value_size, int max_context, const int *position_ptr) {
    constexpr int warp_size = 32;
    constexpr int warps = 4;
    constexpr int q8_1_values = 128;
    constexpr int q8_1_bytes = q8_1_values;
    constexpr int q8_1_blocks = 256 / q8_1_values;
    constexpr float kq_max_offset = 3.0f * 0.6931f;

    const int head = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int tid = warp * warp_size + lane;
    if (head >= heads || head_size != 256 || value_size != 256) return;

    const int position = *position_ptr;
    if (position < 0 || position >= max_context) return;
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const int query_offset = head * 2 * head_size;
    const int key_blocks = head_size / q8_block_width;
    const int value_blocks = value_size / q8_block_width;
    const int key_row_bytes = key_blocks * q8_block_bytes;
    const int value_row_bytes = value_blocks * q8_block_bytes;

    /* llama.cpp quantizes the F32 query into two Q8_1 blocks of 128 values,
     * after applying the attention scale.  One warp handles one 32-value
     * quarter of each Q8_1 block; the four warp maxima are then combined. */
    __shared__ int8_t query_q8[2 * q8_1_bytes];
    __shared__ float query_scales[q8_1_blocks];
    __shared__ float query_max_partials[q8_1_blocks * warps];
    const float query_scale = 1.0f / sqrtf(static_cast<float>(head_size));
    for (int qblock = 0; qblock < q8_1_blocks; ++qblock) {
        const int index = qblock * q8_1_values + warp * warp_size + lane;
        const float scaled = query[query_offset + index] * query_scale;
        float maximum = fabsf(scaled);
        for (int offset = warp_size / 2; offset > 0; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
        if (lane == 0)
            query_max_partials[qblock * warps + warp] = maximum;
        __syncthreads();
        if (tid == qblock) {
            float block_max = query_max_partials[qblock * warps];
            for (int w = 1; w < warps; ++w)
                block_max = fmaxf(block_max, query_max_partials[qblock * warps + w]);
            query_scales[qblock] = block_max / 127.0f;
        }
        __syncthreads();
        const float inverse = query_scales[qblock] != 0.0f ?
            1.0f / query_scales[qblock] : 0.0f;
        const float quantized = scaled * inverse;
        query_q8[qblock * q8_1_values + warp * warp_size + lane] =
            static_cast<int8_t>(roundf(quantized));
        __syncthreads();
    }

    __shared__ float scores[warps * warp_size];
    float value_partial[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        value_partial[i] = 0.0f;
    float running_max = -FLT_MAX / 2.0f;
    float running_sum = 0.0f;
    for (int tile_start = 0; tile_start <= position; tile_start += warps * warp_size) {
        /* Every lane contributes a different dimension slice to each of the
         * 32 keys in this warp, exactly like llama.cpp's nthreads_KQ=32
         * loop.  Keep one score per key for the subsequent value pass. */
        for (int local_key = 0; local_key < warp_size; ++local_key) {
            const int key_index = tile_start + warp * warp_size + local_key;
            const bool valid = key_index <= position && key_index < max_context;
            float score = 0.0f;
            if (valid) {
                const int8_t *key_row = key_cache +
                    (static_cast<size_t>(key_index) * key_value_heads + kv_head) * key_row_bytes;
                const int word = lane & 7;
                for (int qblock = 0; qblock < q8_1_blocks; ++qblock) {
                    const int block = qblock * 4 + (lane >> 3);
                    const int8_t *key_block = key_row + block * q8_block_bytes;
                    const int key_word = load_i32(key_block + 2 + word * 4);
                    const int query_word = load_i32(query_q8 +
                        qblock * q8_1_bytes + lane * 4);
                    const int dot = __dp4a(key_word, query_word, 0);
                    score += block_scale(key_block) * query_scales[qblock] *
                        static_cast<float>(dot);
                }
            }
            for (int offset = warp_size / 2; offset > 0; offset >>= 1)
                score += __shfl_xor_sync(0xffffffffu, score, offset);
            if (lane == local_key)
                scores[warp * warp_size + local_key] = valid ? score : -FLT_MAX / 2.0f;
        }
        __syncwarp();
        const float score = scores[warp * warp_size + lane];
        const bool valid = tile_start + warp * warp_size + lane <= position &&
            tile_start + warp * warp_size + lane < max_context;
        float tile_max = score + kq_max_offset;
        for (int offset = warp_size / 2; offset > 0; offset >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
        const float new_max = fmaxf(running_max, tile_max);
        const float rescale = expf(running_max - new_max);
        const float weight = valid ? expf(score - new_max) : 0.0f;
        running_sum = running_sum * rescale + weight;
        for (int i = 0; i < 8; ++i) value_partial[i] *= rescale;
        /* For the value pass nthreads_V=32: all lanes process one key at a
         * time, each owning four low and four high output components. */
        for (int local_key = 0; local_key < warp_size; ++local_key) {
            const int key_index = tile_start + warp * warp_size + local_key;
            const bool source_valid = key_index <= position && key_index < max_context;
            if (source_valid) {
                const float source_weight = expf(
                    scores[warp * warp_size + local_key] - new_max);
                const int8_t *value_row = value_cache +
                    (static_cast<size_t>(key_index) * key_value_heads + kv_head) * value_row_bytes;
                for (int half = 0; half < 2; ++half) {
                    const int base = half * 128 + lane * 4;
                    const int block = base / q8_block_width;
                    const float scale = block_scale(value_row + block * q8_block_bytes);
                    for (int i = 0; i < 4; ++i) {
                        const int component = base + i;
                        const int offset = component % q8_block_width;
                        const float value = scale * static_cast<float>(
                            value_row[block * q8_block_bytes + 2 + offset]);
                        value_partial[half * 4 + i] += source_weight * value;
                    }
                }
            }
        }
        running_max = new_max;
    }

    __shared__ float warp_max[warps];
    __shared__ float warp_sum[warps];
    __shared__ float partial[warps * 256];
    __shared__ float global_max;
    __shared__ float global_sum;
    float denominator = running_sum;
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        denominator += __shfl_xor_sync(0xffffffffu, denominator, offset);
    if (lane == 0) {
        warp_max[warp] = running_max;
        warp_sum[warp] = denominator;
    }
    __syncthreads();
    if (tid == 0) {
        float maximum = warp_max[0];
        for (int w = 1; w < warps; ++w) maximum = fmaxf(maximum, warp_max[w]);
        global_max = maximum;
        float total = 0.0f;
        for (int w = 0; w < warps; ++w)
            total += warp_sum[w] * expf(warp_max[w] - maximum);
        global_sum = total;
    }
    __syncthreads();
    const float warp_scale = expf(running_max - global_max);
    /* Map each lane's 4 low and 4 high values into a complete 256-element
     * warp partial.  Four float2 registers represent components
     * [lane*4..lane*4+3] and [128+lane*4..]. */
    for (int i = 0; i < 4; ++i) {
        partial[warp * 256 + lane * 4 + i] = value_partial[i] * warp_scale;
        partial[warp * 256 + 128 + lane * 4 + i] = value_partial[4 + i] * warp_scale;
    }
    __syncthreads();
    if (tid < 128) {
        const float *gate = query + query_offset + head_size;
        for (int component = tid; component < value_size; component += 128) {
            float total = 0.0f;
            for (int w = 0; w < warps; ++w)
                total += partial[w * 256 + component];
            attention[head * value_size + component] =
                (global_sum > 0.0f ? total / global_sum : 0.0f) * qwen_sigmoid(gate[component]);
        }
    }
}

static bool qwen_attention_q8_vector_requested() {
    const char *value = std::getenv("FORTAI_CUDA_ATTENTION_Q8_VECTOR");
    return value && (value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
        value[0] == 't' || value[0] == 'T');
}

/* The tiled Q8 GQA kernel is the production default.  Set this to 0 while
 * investigating a device-specific issue to fall back to the bounded-memory
 * one-row online kernel. */
static bool qwen_attention_q8_gqa_batch_enabled() {
    const char *value = std::getenv("FORTAI_CUDA_ATTENTION_Q8_GQA");
    return !value || (value[0] != '0' && value[0] != 'n' && value[0] != 'N' &&
        value[0] != 'f' && value[0] != 'F');
}

/* Tensor-core Q8 attention dequantizes only the active K/V view into a
 * reusable context scratch buffer, matching llama.cpp's temporary F16 pool.
 * It is the default for batched prompts; set FORTAI_CUDA_ATTENTION_Q8_MMA=0
 * to force the bounded-memory dp4a fallback on a device where the temporary
 * view cannot fit. */
static bool qwen_attention_q8_mma_requested() {
    const char *value = std::getenv("FORTAI_CUDA_ATTENTION_Q8_MMA");
    return !value || (value[0] != '0' && value[0] != 'n' && value[0] != 'N' &&
        value[0] != 'f' && value[0] != 'F');
}

/* Convert only the rows participating in the current attention call.  This
 * mirrors llama.cpp's `need_f16_K/V` temporary-pool path: Q8_0 remains the
 * resident cache format, while the MMA kernel consumes a compact, contiguous
 * F16 view.  The view is reused by all attention layers on this CUDA context,
 * so peak memory is proportional to the active prefix rather than duplicated
 * once per layer. */
template <int Width>
__global__ void qwen_attention_dequantize_q8_view(
    const int8_t *__restrict__ source, __half *__restrict__ destination,
    int rows, int key_value_heads) {
    const int row_head = static_cast<int>(blockIdx.x);
    const int token = row_head / key_value_heads;
    const int kv_head = row_head - token * key_value_heads;
    if (token >= rows) return;
    constexpr int blocks = Width / q8_block_width;
    constexpr int row_bytes = blocks * q8_block_bytes;
    const int8_t *source_row = source +
        (static_cast<size_t>(token) * key_value_heads + kv_head) * row_bytes;
    __half *destination_row = destination +
        (static_cast<size_t>(token) * key_value_heads + kv_head) * Width;
    for (int component = static_cast<int>(threadIdx.x);
        component < Width; component += static_cast<int>(blockDim.x)) {
        const int block = component / q8_block_width;
        const int offset = component - block * q8_block_width;
        const int8_t *q8_block = source_row + block * q8_block_bytes;
        destination_row[component] = __float2half(
            block_scale(q8_block) * static_cast<float>(q8_block[2 + offset]));
    }
}

static cudaError_t ensure_attention_f16_view(fortai_cuda_q8_context_impl *context,
    size_t key_bytes, size_t value_bytes) {
    if (!context) return cudaErrorInvalidValue;
    cudaError_t error = cudaSuccess;
    if (key_bytes > context->scratch_attention_key_f16_bytes) {
        __half *replacement = nullptr;
        error = cudaMalloc(reinterpret_cast<void **>(&replacement), key_bytes);
        if (error != cudaSuccess) return error;
        error = cudaFree(context->scratch_attention_key_f16);
        if (error != cudaSuccess) {
            cudaFree(replacement);
            return error;
        }
        context->scratch_attention_key_f16 = replacement;
        context->scratch_attention_key_f16_bytes = key_bytes;
    }
    if (value_bytes > context->scratch_attention_value_f16_bytes) {
        __half *replacement = nullptr;
        error = cudaMalloc(reinterpret_cast<void **>(&replacement), value_bytes);
        if (error != cudaSuccess) return error;
        error = cudaFree(context->scratch_attention_value_f16);
        if (error != cudaSuccess) {
            cudaFree(replacement);
            return error;
        }
        context->scratch_attention_value_f16 = replacement;
        context->scratch_attention_value_f16_bytes = value_bytes;
    }
    return cudaSuccess;
}

static cudaError_t ensure_attention_fattn_scratch(
    fortai_cuda_q8_context_impl *context, size_t mask_bytes, size_t meta_bytes) {
    if (!context) return cudaErrorInvalidValue;
    cudaError_t error = cudaSuccess;
    if (mask_bytes > context->scratch_attention_mask_f16_bytes) {
        __half *replacement = nullptr;
        error = cudaMalloc(reinterpret_cast<void **>(&replacement), mask_bytes);
        if (error != cudaSuccess) return error;
        error = cudaFree(context->scratch_attention_mask_f16);
        if (error != cudaSuccess) {
            cudaFree(replacement);
            return error;
        }
        context->scratch_attention_mask_f16 = replacement;
        context->scratch_attention_mask_f16_bytes = mask_bytes;
    }
    if (meta_bytes > context->scratch_attention_meta_bytes) {
        void *replacement = nullptr;
        error = cudaMalloc(&replacement, meta_bytes);
        if (error != cudaSuccess) return error;
        error = cudaFree(context->scratch_attention_meta);
        if (error != cudaSuccess) {
            cudaFree(replacement);
            return error;
        }
        context->scratch_attention_meta = replacement;
        context->scratch_attention_meta_bytes = meta_bytes;
    }
    return cudaSuccess;
}

/* Exact-reduction vector attention.  The eight-lane key ownership follows
 * llama.cpp, but each lane first forms the four partial sums that its
 * original 32-lane reduction would have paired (0+16, 8+24, then 4,2,1).
 * This cuts the QK shuffle tree from five stages to three without changing a
 * single FP32 addition in the established scalar result.  The value and
 * softmax loops intentionally retain the established order until this path
 * has passed a long token oracle trace. */
__launch_bounds__(128, 1)
__global__ void qwen_attention_apply_f16_exact_vector(const float *__restrict__ query,
    const __half *__restrict__ key_cache, const __half *__restrict__ value_cache,
    float *__restrict__ attention, int heads, int key_value_heads, int head_size,
    int value_size, int max_context, const int *position_ptr) {
    constexpr int warp_size = 32;
    constexpr int warps = 4;
    constexpr int keys_per_warp = warp_size;
    constexpr int keys_per_tile = warps * keys_per_warp;
    constexpr int nthreads_kq = 8;
    constexpr int max_value_parts = 16;
    constexpr float max_offset = 3.0f * 0.6931f;

    const int head = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    if (head >= heads || head_size != 256 || value_size > max_value_parts * warp_size)
        return;

    const int position = *position_ptr;
    const int value_parts = (value_size + warp_size - 1) / warp_size;
    const int group = lane / nthreads_kq;
    const int sublane = lane % nthreads_kq;
    const unsigned subgroup_mask = 0xffu << (group * nthreads_kq);
    const int kv_group = heads / key_value_heads;
    const int kv_head = head / kv_group;
    const int query_offset = head * 2 * head_size;
    const __half scale_h = __float2half_rn(1.0f / sqrtf(static_cast<float>(head_size)));

    /* Match the old full-warp reduction's partial ordering.  The dimension
     * groups are the lane's original slice and its XOR-16/8 partners. */
    const int dimension_group[4] = {
        sublane, sublane + 16, sublane + 8, sublane + 24};
    half2 query_reg[16];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int component = dimension_group[part] * 8;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int index = component + pair * 2;
            query_reg[part * 4 + pair] = make_half2(
                __hmul(__float2half_rn(query[query_offset + index]), scale_h),
                __hmul(__float2half_rn(query[query_offset + index + 1]), scale_h));
        }
    }

    extern __shared__ float shared[];
    float *scores = shared;
    float *warp_maxima = scores + keys_per_tile;
    float *warp_denominators = warp_maxima + warps;
    float *partial_output = warp_denominators + warps;

    float output[max_value_parts] = {};
    float denominator = 0.0f;
    float maximum = -FLT_MAX / 2.0f;

    for (int tile_start = 0; tile_start <= position; tile_start += keys_per_tile) {
#pragma unroll
        for (int key_slot = 0; key_slot < nthreads_kq; ++key_slot) {
            const int local_key = group * nthreads_kq + key_slot;
            const int previous = tile_start + warp * keys_per_warp + local_key;
            const bool valid = previous <= position && previous < max_context;
            float partial[4] = {};
            if (valid) {
                const size_t key_offset =
                    (static_cast<size_t>(previous) * key_value_heads + kv_head) *
                    static_cast<size_t>(head_size);
                const half2 *key_f16 = reinterpret_cast<const half2 *>(key_cache + key_offset);
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    const int key_pair_base = dimension_group[part] * 4;
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 key_value = __half22float2(key_f16[key_pair_base + pair]);
                        const float2 query_value = __half22float2(query_reg[part * 4 + pair]);
                        partial[part] = fmaf(query_value.x, key_value.x, partial[part]);
                        partial[part] = fmaf(query_value.y, key_value.y, partial[part]);
                    }
                }
            }

            /* This grouping is algebraically the same as the old full-warp
             * XOR16/8/4/2/1 tree, including its FP32 rounding points. */
            const float pair0 = partial[0] + partial[1];
            const float pair8 = partial[2] + partial[3];
            float dot = pair0 + pair8;
#pragma unroll
            for (int offset = nthreads_kq / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(subgroup_mask, dot, offset, nthreads_kq);
            if (sublane == key_slot)
                scores[warp * keys_per_warp + local_key] = valid ? dot : -FLT_MAX / 2.0f;
        }
        __syncthreads();

        float tile_maximum = -FLT_MAX / 2.0f;
#pragma unroll
        for (int local_key = 0; local_key < keys_per_warp; ++local_key)
            tile_maximum = fmaxf(tile_maximum,
                scores[warp * keys_per_warp + local_key] + max_offset);
        const float new_maximum = fmaxf(maximum, tile_maximum);
        const float rescale = expf(maximum - new_maximum);
        const int own_key = warp * keys_per_warp + lane;
        const bool own_valid = tile_start + own_key <= position &&
            tile_start + own_key < max_context;
        const float own_weight = own_valid ?
            expf(scores[own_key] - new_maximum) : 0.0f;
        denominator = denominator * rescale + own_weight;

#pragma unroll
        for (int part = 0; part < max_value_parts; ++part)
            if (part < value_parts) output[part] *= rescale;

#pragma unroll
        for (int local_key = 0; local_key < keys_per_warp; ++local_key) {
            const int previous = tile_start + warp * keys_per_warp + local_key;
            if (previous > position || previous >= max_context) continue;
            const float weight = expf(scores[warp * keys_per_warp + local_key] - new_maximum);
            const size_t value_offset =
                (static_cast<size_t>(previous) * key_value_heads + kv_head) *
                static_cast<size_t>(value_size);
#pragma unroll
            for (int part = 0; part < max_value_parts; ++part) {
                const int component = part * warp_size + lane;
                if (part < value_parts && component < value_size)
                    output[part] += weight * __half2float(value_cache[value_offset + component]);
            }
        }
        maximum = new_maximum;
        __syncthreads();
    }

    float warp_denominator = denominator;
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        warp_denominator += __shfl_xor_sync(0xffffffffu, warp_denominator, offset, warp_size);
    if (lane == 0) {
        warp_maxima[warp] = maximum;
        warp_denominators[warp] = warp_denominator;
    }
#pragma unroll
    for (int part = 0; part < max_value_parts; ++part) {
        const int component = part * warp_size + lane;
        if (part < value_parts && component < value_size)
            partial_output[warp * value_size + component] = output[part];
    }
    __syncthreads();

    float global_maximum = -FLT_MAX / 2.0f;
#pragma unroll
    for (int w = 0; w < warps; ++w)
        global_maximum = fmaxf(global_maximum, warp_maxima[w]);
    const float global_scale = expf(maximum - global_maximum);
    (void) global_scale;
    float global_denominator = lane < warps ? warp_denominators[lane] *
        expf(warp_maxima[lane] - global_maximum) : 0.0f;
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        global_denominator += __shfl_xor_sync(0xffffffffu, global_denominator, offset, warp_size);

#pragma unroll
    for (int part = 0; part < max_value_parts; ++part) {
        const int component = part * warp_size + lane;
        if (part >= value_parts || component >= value_size) continue;
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < warps; ++w)
            total += partial_output[w * value_size + component] *
                expf(warp_maxima[w] - global_maximum);
        const float *gate = query + query_offset + head_size;
        attention[head * value_size + component] =
            (global_denominator > 0.0f ? total / global_denominator : 0.0f) *
            qwen_sigmoid(gate[component]);
    }
}

/* Decode-time F16 attention follows llama.cpp's vector kernel layout.  Each
 * warp owns a 32-key slice, keeps its own online-softmax state, and combines
 * the four warp partials only after the key scan.  Besides avoiding the
 * block-wide barrier for every key, this preserves the reduction/rounding
 * order used by the production Qwen3.5 CUDA oracle (Q=F16-scaled, K/V=F16,
 * XOR warp reductions, and the same max offset).
 */
__launch_bounds__(128, 1)
__global__ void qwen_attention_apply_f16_llama(const float *query,
    const __half *key_cache, const __half *value_cache, float *attention,
    int heads, int key_value_heads, int head_size, int value_size,
    int max_context, const int *position_ptr) {
    constexpr int warp_size = 32;
    constexpr int warps = 4;
    constexpr int keys_per_warp = warp_size;
    constexpr int keys_per_tile = warps * keys_per_warp;
    constexpr int max_value_parts = 16;
    constexpr float max_offset = 3.0f * 0.6931f;

    const int head = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / warp_size;
    const int lane = tid & (warp_size - 1);
    if (head >= heads) return;
    const int value_parts = (value_size + warp_size - 1) / warp_size;
    if (value_parts > max_value_parts) return;

    const int position = *position_ptr;
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const int query_offset = head * 2 * head_size;
    const __half scale_h = __float2half_rn(1.0f / sqrtf(static_cast<float>(head_size)));

    /* Q is invariant across the whole key scan.  Convert and scale each
     * lane's eight query components once instead of re-reading/converting
     * them for every cached key.  Keep scalar half operations here so the
     * exact reduction path is unchanged. */
    __half query_scaled[8];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const int component = lane * 8 + pair * 2;
        query_scaled[2 * pair] = __hmul(
            __float2half_rn(query[query_offset + component]), scale_h);
        query_scaled[2 * pair + 1] = __hmul(
            __float2half_rn(query[query_offset + component + 1]), scale_h);
    }

    extern __shared__ float shared[];
    float *scores = shared;
    float *warp_maxima = scores + keys_per_tile;
    float *warp_denominators = warp_maxima + warps;
    float *partial_output = warp_denominators + warps;

    float output[max_value_parts] = {};
    float denominator = 0.0f;
    float maximum = -FLT_MAX / 2.0f;

    for (int tile_start = 0; tile_start <= position; tile_start += keys_per_tile) {
        /* Every lane computes all 32 scores in its warp.  The component
         * grouping matches llama.cpp's four half2 loads per lane for D=256.
         */
#pragma unroll
        for (int local_key = 0; local_key < keys_per_warp; ++local_key) {
            const int previous = tile_start + warp * keys_per_warp + local_key;
            float dot = 0.0f;
            if (previous <= position && previous < max_context) {
                const size_t key_offset =
                    (static_cast<size_t>(previous) * key_value_heads + kv_head) *
                    static_cast<size_t>(head_size);
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const int component = lane * 8 + pair * 2;
                    if (component + 1 < head_size) {
                        const float k0 = __half2float(key_cache[key_offset + component]);
                        const float k1 = __half2float(key_cache[key_offset + component + 1]);
                        dot = fmaf(__half2float(query_scaled[2 * pair]), k0, dot);
                        dot = fmaf(__half2float(query_scaled[2 * pair + 1]), k1, dot);
                    }
                }
            }
#pragma unroll
            for (int offset = warp_size / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, offset, warp_size);
            if (lane == local_key)
                scores[warp * keys_per_warp + local_key] =
                    (previous <= position && previous < max_context) ? dot : -FLT_MAX / 2.0f;
        }
        __syncthreads();

        float tile_maximum = -FLT_MAX / 2.0f;
#pragma unroll
        for (int local_key = 0; local_key < keys_per_warp; ++local_key)
            tile_maximum = fmaxf(tile_maximum,
                scores[warp * keys_per_warp + local_key] + max_offset);
        const float new_maximum = fmaxf(maximum, tile_maximum);
        const float rescale = expf(maximum - new_maximum);
        const int own_key = warp * keys_per_warp + lane;
        const bool own_valid = tile_start + own_key <= position &&
            tile_start + own_key < max_context;
        const float own_weight = own_valid ?
            expf(scores[own_key] - new_maximum) : 0.0f;
        denominator = denominator * rescale + own_weight;

#pragma unroll
        for (int part = 0; part < max_value_parts; ++part)
            if (part < value_parts) output[part] *= rescale;

        /* Keep the key-major/value-minor accumulation order of VKQ in the
         * llama.cpp vector kernel. */
#pragma unroll
        for (int local_key = 0; local_key < keys_per_warp; ++local_key) {
            const int previous = tile_start + warp * keys_per_warp + local_key;
            if (previous > position || previous >= max_context) continue;
            const float weight = expf(scores[warp * keys_per_warp + local_key] - new_maximum);
            const size_t value_offset =
                (static_cast<size_t>(previous) * key_value_heads + kv_head) *
                static_cast<size_t>(value_size);
#pragma unroll
            for (int part = 0; part < max_value_parts; ++part) {
                const int component = part * warp_size + lane;
                if (part < value_parts && component < value_size)
                    output[part] += weight * __half2float(value_cache[value_offset + component]);
            }
        }
        maximum = new_maximum;
        __syncthreads();
    }

    float warp_denominator = denominator;
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        warp_denominator += __shfl_xor_sync(0xffffffffu, warp_denominator, offset, warp_size);
    if (lane == 0) {
        warp_maxima[warp] = maximum;
        warp_denominators[warp] = warp_denominator;
    }
#pragma unroll
    for (int part = 0; part < max_value_parts; ++part) {
        const int component = part * warp_size + lane;
        if (part < value_parts && component < value_size)
            partial_output[warp * value_size + component] = output[part];
    }
    __syncthreads();

    float global_maximum = -FLT_MAX / 2.0f;
#pragma unroll
    for (int w = 0; w < warps; ++w)
        global_maximum = fmaxf(global_maximum, warp_maxima[w]);
    const float local_scale = expf(maximum - global_maximum);
    float global_denominator = lane < warps ? warp_denominators[lane] *
        expf(warp_maxima[lane] - global_maximum) : 0.0f;
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        global_denominator += __shfl_xor_sync(0xffffffffu, global_denominator, offset, warp_size);

#pragma unroll
    for (int part = 0; part < max_value_parts; ++part) {
        const int component = part * warp_size + lane;
        if (part >= value_parts || component >= value_size) continue;
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < warps; ++w)
            total += partial_output[w * value_size + component] *
                expf(warp_maxima[w] - global_maximum);
        const float *gate = query + query_offset + head_size;
        attention[head * value_size + component] =
            (global_denominator > 0.0f ? total / global_denominator : 0.0f) *
            qwen_sigmoid(gate[component]);
    }
}

/* Batched counterpart of the short-context F16 attention kernel above.  Keep
 * the two-pass score/softmax/reduction order identical to the scalar path so
 * prompt batching does not accumulate a different online-softmax rounding
 * error across alternating recurrent/attention blocks.  Long contexts still
 * use the bounded-memory online kernel below. */
__global__ void qwen_attention_apply_f16_batch(const float *query,
    const __half *key_cache, const __half *value_cache, float *attention,
    int heads, int key_value_heads, int head_size, int value_size,
    int max_context, int start_position, int query_stride, int attention_stride,
    int batch) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (head >= heads || token >= batch) return;
    const int position = start_position + token;
    if (position < 0 || position >= max_context) return;
    extern __shared__ float shared[];
    float *warps = shared;
    float *scores = shared + 8;
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const size_t query_offset = static_cast<size_t>(token) * query_stride +
        static_cast<size_t>(head) * 2 * head_size;
    const size_t output_offset = static_cast<size_t>(token) * attention_stride +
        static_cast<size_t>(head) * value_size;
    for (int previous = 0; previous <= position; ++previous) {
        float dot = 0.0f;
        const size_t key_offset = (static_cast<size_t>(previous) * key_value_heads + kv_head) *
            head_size;
        for (int i = tid; i < head_size; i += blockDim.x)
            dot += query[query_offset + i] * __half2float(key_cache[key_offset + i]);
        for (int offset = 16; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        if (lane == 0) warps[warp] = dot;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int i = 0; i < (blockDim.x + 31) / 32; ++i) total += warps[i];
            scores[previous] = total / sqrtf(static_cast<float>(head_size));
        }
        __syncthreads();
    }
    if (tid == 0) {
        float maximum = -FLT_MAX;
        for (int previous = 0; previous <= position; ++previous)
            maximum = fmaxf(maximum, scores[previous]);
        float normalizer = 0.0f;
        for (int previous = 0; previous <= position; ++previous) {
            scores[previous] = expf(scores[previous] - maximum);
            normalizer += scores[previous];
        }
        for (int previous = 0; previous <= position; ++previous)
            scores[previous] /= normalizer;
    }
    __syncthreads();
    const float *gate = query + query_offset + head_size;
    for (int i = tid; i < value_size; i += blockDim.x) {
        float total = 0.0f;
        for (int previous = 0; previous <= position; ++previous) {
            const size_t value_offset = (static_cast<size_t>(previous) * key_value_heads + kv_head) *
                value_size;
            total += scores[previous] * __half2float(value_cache[value_offset + i]);
        }
        attention[output_offset + i] = total * qwen_sigmoid(gate[i]);
    }
}

/* Warp-tiled online softmax for batched F16 attention.  Eight warps process
 * thirty-two keys at a time (four keys per warp); the wider tile amortizes
 * the block barriers that otherwise dominate single-token decode.  Each warp
 * owns four scores and a private value partial, while block-wide max/norm
 * scalars keep the exact causal online-softmax invariant. */
__launch_bounds__(256, 1)
__global__ void qwen_attention_apply_flash_f16_batch(const float *query,
    const __half *key_cache, const __half *value_cache, float *attention,
    int heads, int key_value_heads, int head_size, int value_size,
    int max_context, int start_position, int query_stride, int attention_stride,
    int batch) {
    constexpr int warps = 8;
    constexpr int warp_size = 32;
    constexpr int keys_per_warp = 4;
    constexpr int tile_keys = warps * keys_per_warp;
    constexpr int max_value_parts = 16;
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / warp_size;
    const int lane = tid & (warp_size - 1);
    if (head >= heads || token >= batch) return;
    const int position = start_position + token;
    if (position < 0 || position >= max_context) return;
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const size_t query_offset = static_cast<size_t>(token) * query_stride +
        static_cast<size_t>(head) * 2 * head_size;
    const size_t output_offset = static_cast<size_t>(token) * attention_stride +
        static_cast<size_t>(head) * value_size;
    extern __shared__ float shared[];
    float *scores = shared;
    float *weights = shared + tile_keys;
    float *partials = shared + 2 * tile_keys;
    __shared__ float online_max;
    __shared__ float online_norm;
    __shared__ float online_rescale;
    float accumulator[max_value_parts] = {};
    const int value_parts = (value_size + warp_size - 1) / warp_size;

    if (tid == 0) {
        online_max = -FLT_MAX;
        online_norm = 0.0f;
    }
    __syncthreads();

    for (int tile_start = 0; tile_start <= position; tile_start += tile_keys) {
#pragma unroll
        for (int key_in_warp = 0; key_in_warp < keys_per_warp; ++key_in_warp) {
            const int previous = tile_start + warp * keys_per_warp + key_in_warp;
            float dot = 0.0f;
            if (previous <= position) {
                const size_t key_offset = (static_cast<size_t>(previous) * key_value_heads +
                    kv_head) * head_size;
                for (int i = lane; i < head_size; i += warp_size)
                    dot = fmaf(query[query_offset + i],
                        __half2float(key_cache[key_offset + i]), dot);
            }
            for (int offset = warp_size / 2; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffffu, dot, offset);
            if (lane == 0)
                scores[warp * keys_per_warp + key_in_warp] =
                    previous <= position ? dot / sqrtf(static_cast<float>(head_size)) : -FLT_MAX;
        }
        __syncthreads();

        if (tid == 0) {
            float tile_max = -FLT_MAX;
#pragma unroll
            for (int index = 0; index < tile_keys; ++index)
                tile_max = fmaxf(tile_max, scores[index]);
            const float old_max = online_max;
            const float new_max = fmaxf(old_max, tile_max);
            const float rescale = expf(old_max - new_max);
            float tile_norm = 0.0f;
#pragma unroll
            for (int index = 0; index < tile_keys; ++index) {
                weights[index] = scores[index] == -FLT_MAX ? 0.0f :
                    expf(scores[index] - new_max);
                tile_norm += weights[index];
            }
            online_max = new_max;
            online_norm = online_norm * rescale + tile_norm;
            online_rescale = rescale;
        }
        __syncthreads();

        for (int part = 0; part < max_value_parts; ++part) {
            if (part < value_parts) accumulator[part] *= online_rescale;
        }
#pragma unroll
        for (int key_in_warp = 0; key_in_warp < keys_per_warp; ++key_in_warp) {
            const int previous = tile_start + warp * keys_per_warp + key_in_warp;
            if (previous <= position) {
                const size_t value_offset = (static_cast<size_t>(previous) * key_value_heads +
                    kv_head) * value_size;
#pragma unroll
                for (int part = 0; part < max_value_parts; ++part) {
                    const int index = part * warp_size + lane;
                    if (part < value_parts && index < value_size) {
                        const float value = __half2float(value_cache[value_offset + index]);
                        accumulator[part] += weights[warp * keys_per_warp + key_in_warp] * value;
                    }
                }
            }
        }
        __syncthreads();
    }

    /* Reduce each value lane over the eight key-owning warps. */
#pragma unroll
    for (int part = 0; part < max_value_parts; ++part) {
        if (part < value_parts) {
            const int index = part * warp_size + lane;
            partials[warp * value_size + index] = accumulator[part];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int part = 0; part < max_value_parts; ++part) {
            if (part < value_parts) {
                const int index = part * warp_size + lane;
                float total = 0.0f;
#pragma unroll
                for (int w = 0; w < warps; ++w)
                    total += partials[w * value_size + index];
                const float inverse = online_norm > 0.0f ? 1.0f / online_norm : 0.0f;
                const float *gate = query + query_offset + head_size;
                attention[output_offset + index] = total * inverse * qwen_sigmoid(gate[index]);
            }
        }
    }
}

/* Tensor-core flash attention for the common Qwen head shape.  A block owns
 * one head and sixteen consecutive query tokens.  It computes Q*K^T in four
 * 16x16 MMA tiles for each 16-wide slice of the head, then performs the
 * causal online softmax and V accumulation while the score tile is resident
 * in shared memory.  This is the same two-pass invariant as the scalar tiled
 * kernel above, but the score reduction is matrix-multiply throughput rather
 * than one warp per key. */
__launch_bounds__(256, 1)
__global__ void qwen_attention_apply_mma_f16_batch(const float *__restrict__ query,
    const __half *__restrict__ key_cache, const __half *__restrict__ value_cache,
    float *__restrict__ attention,
    int heads, int key_value_heads, int head_size, int value_size,
    int max_context, int start_position, int query_stride, int attention_stride,
    int batch, int query_head_stride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    constexpr int warp_size = 32;
    constexpr int query_tile = 16;
    constexpr int key_tile = 64;
    constexpr int key_mma_tile = 16;
    constexpr int max_value_parts = 16;
    const int head = static_cast<int>(blockIdx.x);
    const int query_tile_index = static_cast<int>(blockIdx.y);
    const int query_base = query_tile_index * query_tile;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / warp_size;
    const int lane = tid & (warp_size - 1);
    if (head >= heads || query_base >= batch) return;

    const int query_count = min(query_tile, batch - query_base);
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const int q_stride = ((head_size + key_mma_tile - 1) / key_mma_tile) * key_mma_tile;
    const int query_head_offset = head * query_head_stride;
    const int max_query_position = min(max_context - 1,
        start_position + query_base + query_count - 1);
    const int key_tiles = max_query_position >= 0 ?
        (max_query_position + key_tile) / key_tile : 0;
    const int value_parts = (value_size + warp_size - 1) / warp_size;

    extern __shared__ unsigned char storage[];
    auto *query_shared = reinterpret_cast<__half *>(storage);
    auto *key_shared = query_shared + query_tile * q_stride;
    auto *scores = reinterpret_cast<float *>(key_shared + key_tile * key_mma_tile);
    auto *row_max = scores + query_tile * key_tile;
    auto *row_norm = row_max + query_tile;
    auto *row_rescale = row_norm + query_tile;
    auto *weights_half = reinterpret_cast<__half *>(row_rescale + query_tile);
    auto *value_shared = weights_half + query_tile * key_tile;
    auto *tile_output = reinterpret_cast<float *>(value_shared + 8 * key_mma_tile * key_mma_tile);
    auto *output_shared = tile_output + 8 * key_mma_tile * key_mma_tile;

    for (int index = tid; index < query_tile * q_stride; index += blockDim.x) {
        const int q = index / q_stride;
        const int k = index - q * q_stride;
        query_shared[index] = (q < query_count && k < head_size) ?
            __float2half(query[static_cast<size_t>(query_base + q) * query_stride +
                query_head_offset + k]) : __float2half(0.0f);
    }
    if (tid < query_tile) {
        row_max[tid] = -FLT_MAX;
        row_norm[tid] = 0.0f;
        row_rescale[tid] = 1.0f;
    }
    __syncthreads();

    for (int index = tid; index < query_tile * value_size; index += blockDim.x)
        output_shared[index] = 0.0f;
    __syncthreads();
    for (int key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
        const int key_base = key_tile_index * key_tile;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_accumulator;
        wmma::fill_fragment(score_accumulator, 0.0f);

        for (int k0 = 0; k0 < head_size; k0 += key_mma_tile) {
            for (int index = tid; index < key_tile * key_mma_tile; index += blockDim.x) {
                const int key = index / key_mma_tile;
                const int k = index - key * key_mma_tile;
                const int source_key = key_base + key;
                key_shared[index] = (source_key < max_context && k0 + k < head_size) ?
                    key_cache[(static_cast<size_t>(source_key) * key_value_heads + kv_head) *
                        head_size + k0 + k] : __float2half(0.0f);
            }
            __syncthreads();

            if (warp < key_tile / key_mma_tile) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, query_shared + k0, q_stride);
                wmma::load_matrix_sync(b, key_shared + warp * key_mma_tile * key_mma_tile,
                    key_mma_tile);
                wmma::mma_sync(score_accumulator, a, b, score_accumulator);
            }
            __syncthreads();
        }
        if (warp < key_tile / key_mma_tile) {
            /* Keep one score tile per key warp.  The row-major store uses a
             * leading dimension of 64 so the four warps are disjoint. */
            wmma::store_matrix_sync(scores + warp * key_mma_tile, score_accumulator,
                key_tile, wmma::mem_row_major);
        }
        __syncthreads();

        /* Apply causal masking and advance the online softmax.  One lane per
         * query row owns the scalar expf work; all lanes then consume the
         * resulting weights for their value coordinates. */
        if (tid < query_tile) {
            const int q = tid;
            const int position = start_position + query_base + q;
            float tile_max = -FLT_MAX;
            for (int key = 0; key < key_tile; ++key) {
                const int source_key = key_base + key;
                float score = scores[q * key_tile + key];
                if (q >= query_count || source_key > position || source_key >= max_context)
                    score = -FLT_MAX;
                score /= sqrtf(static_cast<float>(head_size));
                scores[q * key_tile + key] = score;
                tile_max = fmaxf(tile_max, score);
            }
            const float old_max = row_max[q];
            const float new_max = fmaxf(old_max, tile_max);
            const float rescale = expf(old_max - new_max);
            float tile_norm = 0.0f;
            for (int key = 0; key < key_tile; ++key) {
                const float weight = scores[q * key_tile + key] == -FLT_MAX ? 0.0f :
                    expf(scores[q * key_tile + key] - new_max);
                scores[q * key_tile + key] = weight;
                tile_norm += weight;
            }
            row_max[q] = new_max;
            row_norm[q] = row_norm[q] * rescale + tile_norm;
            row_rescale[q] = rescale;
        }
        __syncthreads();

        for (int index = tid; index < query_tile * key_tile; index += blockDim.x)
            weights_half[index] = __float2half(scores[index]);
        __syncthreads();

        /* V is evaluated with the same 16x16 MMA tile as KQ.  Eight warps
         * process eight value tiles at a time; a 16x16 float accumulator is
         * written to a private shared tile, then rescaled by the per-query
         * online-softmax factor before being merged into the persistent
         * output tile. */
        for (int value_base = 0; value_base < value_size; value_base += 8 * key_mma_tile) {
            const int value_tile = value_base + warp * key_mma_tile;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> value_accumulator;
            wmma::fill_fragment(value_accumulator, 0.0f);
            for (int key_group = 0; key_group < key_tile / key_mma_tile; ++key_group) {
                for (int index = lane; index < key_mma_tile * key_mma_tile; index += warp_size) {
                    const int value = index / key_mma_tile;
                    const int key = index - value * key_mma_tile;
                    const int source_key = key_base + key_group * key_mma_tile + key;
                    const int source_value = value_tile + value;
                    value_shared[warp * key_mma_tile * key_mma_tile + index] =
                        (source_key < max_context && source_value < value_size) ?
                        value_cache[(static_cast<size_t>(source_key) * key_value_heads + kv_head) *
                            value_size + source_value] : __float2half(0.0f);
                }
                __syncwarp();
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, weights_half + key_group * key_mma_tile, key_tile);
                wmma::load_matrix_sync(b, value_shared + warp * key_mma_tile * key_mma_tile,
                    key_mma_tile);
                wmma::mma_sync(value_accumulator, a, b, value_accumulator);
            }
            wmma::store_matrix_sync(tile_output + warp * key_mma_tile * key_mma_tile,
                value_accumulator, key_mma_tile, wmma::mem_row_major);
            __syncwarp();
            for (int index = lane; index < query_tile * key_mma_tile; index += warp_size) {
                const int q = index / key_mma_tile;
                const int value = index - q * key_mma_tile;
                const int source_value = value_tile + value;
                if (source_value < value_size)
                    output_shared[q * value_size + source_value] =
                        output_shared[q * value_size + source_value] * row_rescale[q] +
                        tile_output[warp * key_mma_tile * key_mma_tile + index];
            }
        }
        __syncthreads();
    }

    if (warp < 8) {
        const int q0 = warp * 2;
        const int q1 = q0 + 1;
        const float *gate0 = query + static_cast<size_t>(query_base + q0) * query_stride +
            query_head_offset + head_size;
        const float *gate1 = query + static_cast<size_t>(query_base + q1) * query_stride +
            query_head_offset + head_size;
        for (int part = 0; part < max_value_parts; ++part) {
            const int index = part * warp_size + lane;
            if (part >= value_parts || index >= value_size) continue;
            if (q0 < query_count) {
                const float inverse = row_norm[q0] > 0.0f ? 1.0f / row_norm[q0] : 0.0f;
                attention[static_cast<size_t>(query_base + q0) * attention_stride +
                    head * value_size + index] = output_shared[q0 * value_size + index] * inverse *
                    qwen_sigmoid(gate0[index]);
            }
            if (q1 < query_count) {
                const float inverse = row_norm[q1] > 0.0f ? 1.0f / row_norm[q1] : 0.0f;
                attention[static_cast<size_t>(query_base + q1) * attention_stride +
                    head * value_size + index] = output_shared[q1 * value_size + index] * inverse *
                    qwen_sigmoid(gate1[index]);
            }
        }
    }
#else
    (void) query; (void) key_cache; (void) value_cache; (void) attention;
    (void) heads; (void) key_value_heads; (void) head_size; (void) value_size;
    (void) max_context; (void) start_position; (void) query_stride;
    (void) attention_stride; (void) batch; (void) query_head_stride;
#endif
}

/* Q8_0/Q8_0 batched flash attention for the production Qwen3.5 GQA shape.
 *
 * This follows the useful parts of llama.cpp's fattn-vec path
 * (ggml/src/ggml-cuda/fattn-common.cuh and fattn-vec.cuh): scale the query
 * before Q8_1 quantization, use dp4a for each 32-value Q8 block, keep the
 * softmax online, and accumulate values without a context-sized temporary.
 * The implementation is intentionally native to FortAI rather than a copy
 * of llama.cpp: one block owns a KV head, four query heads, and four adjacent
 * prompt tokens.  A packed K/V tile is staged into shared memory once and reused by
 * all Group*QueryTokens (head,token) attention rows.  This removes the redundant GQA
 * and prompt-token global loads that made the old one-block-per-row kernel
 * fall behind at long prompts while retaining the Q8 cache footprint.
 */
template <int Group, int HeadSize, int ValueSize, int QueryTokens = 4, int KeyTile = 32>
__launch_bounds__(Group * 32, 1)
__global__ void qwen_attention_apply_gqa_q8_batch(
    const float *__restrict__ query,
    const int8_t *__restrict__ key_cache_q8,
    const int8_t *__restrict__ value_cache_q8,
    float *__restrict__ attention,
    int heads, int key_value_heads, int max_context, int start_position,
    int query_stride, int attention_stride, int batch,
    const int *position_ptr = nullptr, float *partial_output = nullptr,
    int max_partitions = 1) {
    constexpr int warp_size = 32;
    constexpr int q8_query_blocks = HeadSize / q8_block_width;
    constexpr int q8_query_groups = q8_query_blocks / 4;
    constexpr int key_row_bytes = (HeadSize / q8_block_width) * q8_block_bytes;
    constexpr int value_row_bytes = (ValueSize / q8_block_width) * q8_block_bytes;
    const float attention_scale = rsqrtf(static_cast<float>(HeadSize));
    constexpr float kq_max_offset = 3.0f * 0.6931f;

    const int kv_head = static_cast<int>(blockIdx.x);
    const int query_base = static_cast<int>(blockIdx.y) * QueryTokens;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / warp_size;
    const int lane = tid & (warp_size - 1);
    if (kv_head >= key_value_heads || Group * key_value_heads != heads ||
        HeadSize != 256 || ValueSize != 256 || query_base >= batch)
        return;

    const int query_count = min(QueryTokens, batch - query_base);
    const int position_base = position_ptr ? *position_ptr : start_position;
    const int max_query_position = min(max_context - 1,
        position_base + query_base + query_count - 1);
    const int partition_count = max_partitions > 1 && max_query_position >= 512 ?
        max_partitions : 1;
    const int partition = static_cast<int>(blockIdx.z);
    if (partition >= partition_count) return;
    const int query_head = kv_head * Group + warp;
    if (warp >= Group) return;

    extern __shared__ unsigned char storage[];
    auto *query_q8 = reinterpret_cast<int8_t *>(storage);
    auto *query_scales = reinterpret_cast<float *>(query_q8 + Group * QueryTokens * HeadSize);
    auto *key_tile = reinterpret_cast<int8_t *>(query_scales + Group * QueryTokens * q8_query_blocks);
    auto *value_tile = key_tile + KeyTile * key_row_bytes;
    auto *scores = reinterpret_cast<float *>(value_tile + KeyTile * value_row_bytes);

    const int query_head_offset = query_head * 2 * HeadSize;

    /* Quantize every prompt query once.  This is the same Q8_1 contract used
     * by llama.cpp's vec_dot_fattn_vec_KQ_q8_0, but the quantized rows are
     * shared by the four query heads and four prompt tokens in this block. */
    for (int q = 0; q < QueryTokens; ++q) {
        const bool valid_query = q < query_count;
        const size_t query_offset = static_cast<size_t>(query_base + q) * query_stride +
            static_cast<size_t>(query_head_offset);
        for (int qgroup = 0; qgroup < q8_query_groups; ++qgroup) {
            const int qblock = qgroup * 4 + (lane >> 3);
            const int base = qblock * q8_block_width + (lane & 7) * 4;
            float maximum = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float value = valid_query ? query[query_offset + base + i] * attention_scale : 0.0f;
                maximum = fmaxf(maximum, fabsf(value));
            }
            for (int offset = 4; offset > 0; offset >>= 1)
                maximum = fmaxf(maximum,
                    __shfl_xor_sync(0xffffffffu, maximum, offset, 8));
            if ((lane & 7) == 0)
                query_scales[(warp * QueryTokens + q) * q8_query_blocks + qblock] = maximum / 127.0f;
            __syncwarp();
            const float scale = query_scales[(warp * QueryTokens + q) * q8_query_blocks + qblock];
            const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float value = valid_query ? query[query_offset + base + i] * attention_scale : 0.0f;
                query_q8[(warp * QueryTokens + q) * HeadSize + base + i] =
                    static_cast<int8_t>(roundf(value * inverse));
            }
            __syncwarp();
        }
    }
    __syncthreads();

    float accumulator[QueryTokens][ValueSize / warp_size];
    float running_max[QueryTokens];
    float running_norm[QueryTokens];
#pragma unroll
    for (int q = 0; q < QueryTokens; ++q) {
        running_max[q] = -FLT_MAX / 2.0f;
        running_norm[q] = 0.0f;
#pragma unroll
        for (int i = 0; i < ValueSize / warp_size; ++i)
            accumulator[q][i] = 0.0f;
    }

    const int key_tile_count = max_query_position >= 0 ?
        (max_query_position + KeyTile) / KeyTile : 0;
    const int first_key_tile = key_tile_count * partition / partition_count;
    const int last_key_tile = key_tile_count * (partition + 1) / partition_count;
    for (int key_tile_index = first_key_tile; key_tile_index < last_key_tile;
        ++key_tile_index) {
        const int key_base = key_tile_index * KeyTile;
        /* All block threads cooperate on one contiguous Q8 K/V tile.  Invalid
         * rows are zeroed; causal masking below still excludes them from the
         * softmax, so the shared tile has no context-sized allocation. */
        /* Stage K and V with 16-byte vector copies.  A byte-at-a-time loop
         * issues one memory transaction per element; at 16k context this
         * kernel moved only 12% of peak bandwidth while accounting for 82% of
         * decode GPU time.  llama.cpp's fattn kernels copy K/V in 16-byte
         * units for the same reason (ggml_cuda_memcpy_1<cpy_nb>). */
        constexpr int copy_bytes = static_cast<int>(sizeof(int4));
        static_assert(key_row_bytes % copy_bytes == 0, "key row must be 16-byte divisible");
        static_assert(value_row_bytes % copy_bytes == 0, "value row must be 16-byte divisible");
        constexpr int key_chunks = key_row_bytes / copy_bytes;
        constexpr int value_chunks = value_row_bytes / copy_bytes;
        for (int chunk = tid; chunk < KeyTile * key_chunks; chunk += blockDim.x) {
            const int key = chunk / key_chunks;
            const int source = key_base + key;
            int4 packed = make_int4(0, 0, 0, 0);
            if (source <= max_query_position) {
                const int4 *row = reinterpret_cast<const int4 *>(key_cache_q8 +
                    (static_cast<size_t>(source) * key_value_heads + kv_head) *
                        static_cast<size_t>(key_row_bytes));
                packed = row[chunk - key * key_chunks];
            }
            reinterpret_cast<int4 *>(key_tile)[chunk] = packed;
        }
        for (int chunk = tid; chunk < KeyTile * value_chunks; chunk += blockDim.x) {
            const int key = chunk / value_chunks;
            const int source = key_base + key;
            int4 packed = make_int4(0, 0, 0, 0);
            if (source <= max_query_position) {
                const int4 *row = reinterpret_cast<const int4 *>(value_cache_q8 +
                    (static_cast<size_t>(source) * key_value_heads + kv_head) *
                        static_cast<size_t>(value_row_bytes));
                packed = row[chunk - key * value_chunks];
            }
            reinterpret_cast<int4 *>(value_tile)[chunk] = packed;
        }
        __syncthreads();

        /* The grouped query heads share this KV row but retain independent Q8 query
         * scales.  Lane 0 stores one complete score per key/query row.
         * Each key's warp reduction is independent of the others, so unrolling
         * lets several of them overlap instead of paying the full shuffle
         * latency once per key; at 16k context this loop was 82% of decode GPU
         * time while moving only 12% of peak memory bandwidth. */
#pragma unroll 4
        for (int local_key = 0; local_key < KeyTile; ++local_key) {
            const int source = key_base + local_key;
            const bool key_in_context = source <= max_query_position;
            int key_words[q8_query_groups];
            float key_scales[q8_query_groups];
#pragma unroll
            for (int qgroup = 0; qgroup < q8_query_groups; ++qgroup) {
                const int block = qgroup * 4 + (lane >> 3);
                const int word = lane & 7;
                const int8_t *key_block = key_tile + local_key * key_row_bytes + block * q8_block_bytes;
                key_words[qgroup] = load_i32(key_block + 2 + word * 4);
                key_scales[qgroup] = block_scale(key_block);
            }
            for (int q = 0; q < QueryTokens; ++q) {
                float dot = 0.0f;
                const int query_index = (warp * QueryTokens + q) * HeadSize;
                const int scale_index = (warp * QueryTokens + q) * q8_query_blocks;
#pragma unroll
                for (int qgroup = 0; qgroup < q8_query_groups; ++qgroup) {
                    const int block = qgroup * 4 + (lane >> 3);
                    const int query_word = load_i32(query_q8 + query_index +
                        block * q8_block_width + (lane & 7) * 4);
                    dot += key_scales[qgroup] * query_scales[scale_index + block] *
                        static_cast<float>(__dp4a(key_words[qgroup], query_word, 0));
                }
                for (int offset = warp_size / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffffu, dot, offset);
                if (lane == 0)
                    scores[(warp * QueryTokens + q) * KeyTile + local_key] =
                        key_in_context ? dot : -FLT_MAX / 2.0f;
            }
        }
        __syncwarp();

        float tile_new_max[QueryTokens];
        float tile_rescale[QueryTokens];
#pragma unroll
        for (int q = 0; q < QueryTokens; ++q) {
            const int position = position_base + query_base + q;
            float tile_max = -FLT_MAX / 2.0f;
#pragma unroll
            for (int local_key = lane; local_key < KeyTile; local_key += warp_size) {
                const int source = key_base + local_key;
                const bool valid = q < query_count && source <= position &&
                    source <= max_query_position;
                const float score = scores[(warp * QueryTokens + q) * KeyTile + local_key];
                tile_max = fmaxf(tile_max,
                    valid ? score + kq_max_offset : -FLT_MAX / 2.0f);
            }
            for (int offset = warp_size / 2; offset > 0; offset >>= 1)
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
            tile_new_max[q] = fmaxf(running_max[q], tile_max);
            tile_rescale[q] = expf(running_max[q] - tile_new_max[q]);
            float tile_norm = 0.0f;
#pragma unroll
            for (int local_key = lane; local_key < KeyTile; local_key += warp_size) {
                const int source = key_base + local_key;
                const bool valid = q < query_count && source <= position &&
                    source <= max_query_position;
                const float score = scores[(warp * QueryTokens + q) * KeyTile + local_key];
                tile_norm += valid ? expf(score - tile_new_max[q]) : 0.0f;
            }
            for (int offset = warp_size / 2; offset > 0; offset >>= 1)
                tile_norm += __shfl_xor_sync(0xffffffffu, tile_norm, offset);
            running_norm[q] = running_norm[q] * tile_rescale[q] + tile_norm;
#pragma unroll
            for (int i = 0; i < ValueSize / warp_size; ++i)
                accumulator[q][i] *= tile_rescale[q];
        }

        /* Every lane owns a set of output components, but must visit all keys
         * in the tile for those components.  Restricting this loop to `lane`
         * would only accumulate one key modulo KeyTile and is an easy-to-miss
         * error because the resulting kernel is much faster while its logits
         * are wrong.  Decode the value scale once per key/lane and reuse it
         * for all query rows in this tile. */
        const int value_base = (lane * (ValueSize / warp_size)) % q8_block_width;
        const int value_block_index = (lane * (ValueSize / warp_size)) / q8_block_width;
        for (int local_key = 0; local_key < KeyTile; ++local_key) {
                const int source = key_base + local_key;
                if (source >= max_context) continue;
                const int8_t *value_row = value_tile + local_key * value_row_bytes;
                const int8_t *value_block = value_row + value_block_index * q8_block_bytes;
                const float value_scale = block_scale(value_block);
#pragma unroll
                for (int q = 0; q < QueryTokens; ++q) {
                    const int position = position_base + query_base + q;
                    if (q >= query_count || source > position) continue;
                    const float weight = expf(
                        scores[(warp * QueryTokens + q) * KeyTile + local_key] - tile_new_max[q]);
#pragma unroll
                    for (int i = 0; i < ValueSize / warp_size; ++i) {
                        accumulator[q][i] += weight * value_scale *
                            static_cast<float>(value_block[2 + value_base + i]);
                    }
                }
        }
#pragma unroll
        for (int q = 0; q < QueryTokens; ++q)
            running_max[q] = tile_new_max[q];
        __syncthreads();
    }

    for (int q = 0; q < QueryTokens; ++q) {
        if (q >= query_count) continue;
        const size_t output_offset = static_cast<size_t>(query_base + q) *
            attention_stride + static_cast<size_t>(query_head) * ValueSize;
        const size_t partial_offset =
            ((static_cast<size_t>(query_base + q) * heads + query_head) *
                max_partitions + partition) * (ValueSize + 2);
        if (partial_output && lane == 0) {
            partial_output[partial_offset] = running_max[q];
            partial_output[partial_offset + 1] = running_norm[q];
        }
#pragma unroll
        for (int i = 0; i < ValueSize / warp_size; ++i) {
            const int component = lane * (ValueSize / warp_size) + i;
            if (partial_output) {
                partial_output[partial_offset + 2 + component] = accumulator[q][i];
            } else {
                const float inverse = running_norm[q] > 0.0f ?
                    1.0f / running_norm[q] : 0.0f;
                const float *gate = query + static_cast<size_t>(query_base + q) *
                    query_stride + query_head_offset + HeadSize;
                attention[output_offset + component] = accumulator[q][i] * inverse *
                    qwen_sigmoid(gate[component]);
            }
        }
    }
}

template <int ValueSize>
__global__ void qwen_attention_combine_q8_partitions(
    const float *__restrict__ query, const float *__restrict__ partial_output,
    float *__restrict__ attention, int heads, int batch, int query_stride,
    int attention_stride, int query_head_stride, const int *position_ptr,
    int max_partitions) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int component = static_cast<int>(threadIdx.x);
    if (head >= heads || token >= batch || component >= ValueSize) return;
    const int max_query_position = *position_ptr + batch - 1;
    const int partition_count = max_partitions > 1 && max_query_position >= 512 ?
        max_partitions : 1;
    const size_t row = static_cast<size_t>(token) * heads + head;
    float maximum = -FLT_MAX / 2.0f;
#pragma unroll
    for (int partition = 0; partition < 24; ++partition) {
        if (partition < partition_count) {
            const size_t offset = (row * max_partitions + partition) *
                (ValueSize + 2);
            maximum = fmaxf(maximum, partial_output[offset]);
        }
    }
    float normalizer = 0.0f;
    float total = 0.0f;
#pragma unroll
    for (int partition = 0; partition < 24; ++partition) {
        if (partition < partition_count) {
            const size_t offset = (row * max_partitions + partition) *
                (ValueSize + 2);
            const float scale = expf(partial_output[offset] - maximum);
            normalizer += partial_output[offset + 1] * scale;
            total += partial_output[offset + 2 + component] * scale;
        }
    }
    const float inverse = normalizer > 0.0f ? 1.0f / normalizer : 0.0f;
    const float gate = query[static_cast<size_t>(token) * query_stride +
        static_cast<size_t>(head) * query_head_stride + ValueSize + component];
    attention[static_cast<size_t>(token) * attention_stride +
        static_cast<size_t>(head) * ValueSize + component] =
        total * inverse * qwen_sigmoid(gate);
}

/* GQA flash attention for the Qwen3.5 production shape (4 query heads per KV
 * head, 256-wide K/V).  The legacy batched kernels assign a CUDA block to one
 * query head and one token, which reloads the same K/V rows eight times.  This
 * kernel assigns one warp to each of the four query heads and four adjacent
 * tokens to the block.  K and V rows are staged once in shared memory and the
 * online-softmax state stays in registers, preserving bounded memory while
 * removing the redundant GQA traffic and 16x launch fan-out. */
template <int Group, int HeadSize, int ValueSize, int QueryTokens = 4, int KeyTile = 64>
__launch_bounds__(128, 1)
__global__ void qwen_attention_apply_gqa_f16_batch(const float *__restrict__ query,
    const __half *__restrict__ key_cache, const __half *__restrict__ value_cache,
    float *__restrict__ attention, int heads, int key_value_heads, int max_context,
    int start_position, int query_stride, int attention_stride, int batch,
    const int *position_ptr = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    constexpr int warp_size = 32;
    constexpr int head_parts = HeadSize / warp_size;
    constexpr int value_parts = ValueSize / warp_size;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int query_base = static_cast<int>(blockIdx.y) * QueryTokens;
    const int warp = static_cast<int>(threadIdx.x) / warp_size;
    const int lane = static_cast<int>(threadIdx.x) & (warp_size - 1);
    if (kv_head >= key_value_heads || warp >= Group ||
        kv_head * Group + warp >= heads || query_base >= batch)
        return;

    const int head = kv_head * Group + warp;
    const int query_count = min(QueryTokens, batch - query_base);
    const int position_base = position_ptr ? *position_ptr : start_position;
    const int max_query_position = min(max_context - 1,
        position_base + query_base + query_count - 1);
    const int key_tiles = max_query_position >= 0 ?
        (max_query_position + KeyTile) / KeyTile : 0;
    const size_t query_head_offset = static_cast<size_t>(head) * 2 * HeadSize;

    extern __shared__ __half gqa_storage[];
    __half *key_tile = gqa_storage;
    __half *value_tile = key_tile + KeyTile * HeadSize;

    float query_reg[QueryTokens][head_parts];
    float output_reg[QueryTokens][value_parts] = {};
    float row_max[QueryTokens];
    float row_norm[QueryTokens] = {};
#pragma unroll
    for (int q = 0; q < QueryTokens; ++q) {
        row_max[q] = -FLT_MAX;
#pragma unroll
        for (int part = 0; part < head_parts; ++part) {
            const int index = part * warp_size + lane;
            query_reg[q][part] = q < query_count ?
                query[static_cast<size_t>(query_base + q) * query_stride +
                    query_head_offset + index] : 0.0f;
        }
    }

    for (int key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
        const int key_base = key_tile_index * KeyTile;
        for (int index = static_cast<int>(threadIdx.x);
            index < KeyTile * HeadSize; index += blockDim.x) {
            const int key = index / HeadSize;
            const int component = index - key * HeadSize;
            const int source_key = key_base + key;
            const bool valid = source_key < max_context;
            key_tile[index] = valid ? key_cache[
                (static_cast<size_t>(source_key) * key_value_heads + kv_head) * HeadSize +
                component] : __float2half(0.0f);
            value_tile[index] = valid ? value_cache[
                (static_cast<size_t>(source_key) * key_value_heads + kv_head) * ValueSize +
                component] : __float2half(0.0f);
        }
        __syncthreads();

        for (int key = 0; key < KeyTile; ++key) {
            const int source_key = key_base + key;
            float dot[QueryTokens] = {};
#pragma unroll
            for (int part = 0; part < head_parts; ++part) {
                const float key_value = __half2float(key_tile[key * HeadSize +
                    part * warp_size + lane]);
#pragma unroll
                for (int q = 0; q < QueryTokens; ++q)
                    dot[q] = fmaf(query_reg[q][part], key_value, dot[q]);
            }
#pragma unroll
            for (int q = 0; q < QueryTokens; ++q) {
#pragma unroll
                for (int offset = warp_size / 2; offset > 0; offset >>= 1)
                    dot[q] += __shfl_down_sync(0xffffffffu, dot[q], offset);
            }

            float weight[QueryTokens];
            float rescale[QueryTokens];
            if (lane == 0) {
#pragma unroll
                for (int q = 0; q < QueryTokens; ++q) {
                    const int position = position_base + query_base + q;
                    const bool valid = q < query_count && source_key <= position &&
                        source_key < max_context;
                    if (!valid) {
                        weight[q] = 0.0f;
                        rescale[q] = 1.0f;
                    } else {
                        const float score = dot[q] / sqrtf(static_cast<float>(HeadSize));
                        const float new_max = fmaxf(row_max[q], score);
                        rescale[q] = expf(row_max[q] - new_max);
                        weight[q] = expf(score - new_max);
                        row_max[q] = new_max;
                        row_norm[q] = row_norm[q] * rescale[q] + weight[q];
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < QueryTokens; ++q) {
                const float w = __shfl_sync(0xffffffffu, weight[q], 0);
                const float r = __shfl_sync(0xffffffffu, rescale[q], 0);
#pragma unroll
                for (int part = 0; part < value_parts; ++part) {
                    const float value = __half2float(value_tile[key * ValueSize +
                        part * warp_size + lane]);
                    output_reg[q][part] = output_reg[q][part] * r + w * value;
                }
            }
        }
        __syncthreads();
    }

    float inverse[QueryTokens];
    if (lane == 0) {
#pragma unroll
        for (int q = 0; q < QueryTokens; ++q)
            inverse[q] = row_norm[q] > 0.0f ? 1.0f / row_norm[q] : 0.0f;
    }
#pragma unroll
    for (int q = 0; q < QueryTokens; ++q) {
        const float inv = __shfl_sync(0xffffffffu, inverse[q], 0);
        if (q >= query_count) continue;
        const int query_index = query_base + q;
        const float *gate = query + static_cast<size_t>(query_index) * query_stride +
            query_head_offset + HeadSize;
#pragma unroll
        for (int part = 0; part < value_parts; ++part) {
            const int index = part * warp_size + lane;
            attention[static_cast<size_t>(query_index) * attention_stride +
                head * ValueSize + index] = output_reg[q][part] * inv * qwen_sigmoid(gate[index]);
        }
    }
#else
    (void) query; (void) key_cache; (void) value_cache; (void) attention;
    (void) heads; (void) key_value_heads; (void) max_context; (void) start_position;
    (void) query_stride; (void) attention_stride; (void) batch;
#endif
}

/* Load one component from either the resident half cache or a Q8_0 cache.
 * Keeping this as a compile-time choice lets the tensor-core kernel use the
 * same WMMA implementation for both cache formats without allocating a
 * second context-sized half cache.  The Q8 path is the same 32-value block
 * layout used by llama.cpp's dequantize_V_q8_0 helper. */
template <bool Quantized, int Width>
__device__ __forceinline__ __half qwen_attention_load_cache_component(
    const void *cache, int source, int kv_head, int key_value_heads, int component) {
    if constexpr (!Quantized) {
        const auto *half_cache = static_cast<const __half *>(cache);
        return half_cache[(static_cast<size_t>(source) * key_value_heads + kv_head) * Width + component];
    } else {
        constexpr int row_bytes = (Width / q8_block_width) * q8_block_bytes;
        const int block = component / q8_block_width;
        const int offset = component - block * q8_block_width;
        const auto *row = static_cast<const int8_t *>(cache) +
            (static_cast<size_t>(source) * key_value_heads + kv_head) * row_bytes;
        const auto *q8_block = row + block * q8_block_bytes;
        return __float2half(block_scale(q8_block) * static_cast<float>(q8_block[2 + offset]));
    }
}

/* Tensor-core grouped flash attention for the Qwen3.5 production shape.  One
 * warp owns each query head and sixteen query tokens for one KV head.  Q/K
 * are multiplied in 16x16x16 WMMA tiles, all query heads share the
 * staged K tile, and the normalized weights are reused for the V pass.  The
 * output tiles are accumulated directly in the strided output rows; each warp
 * owns its four query rows at the final normalization/gate step. */
template <int Group, int HeadSize, int ValueSize, int QueryTokens = 16, int KeyTile = 64,
    bool QuantizedCache = false>
__launch_bounds__(Group * 32, 1)
__global__ void qwen_attention_apply_gqa_mma_f16_batch(const float *__restrict__ query,
    const void *__restrict__ key_cache, const void *__restrict__ value_cache,
    float *__restrict__ attention, int heads, int key_value_heads, int max_context,
    int start_position, int query_stride, int attention_stride, int batch,
    int query_head_stride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    constexpr int warp_size = 32;
    constexpr int key_mma_tile = 16;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int query_base = static_cast<int>(blockIdx.y) * QueryTokens;
    const int warp = static_cast<int>(threadIdx.x) / warp_size;
    const int lane = static_cast<int>(threadIdx.x) & (warp_size - 1);
    if (kv_head >= key_value_heads || warp >= Group || query_base >= batch) return;

    const int head = kv_head * Group + warp;
    if (head >= heads) return;
    const int query_count = min(QueryTokens, batch - query_base);
    const int max_query_position = min(max_context - 1,
        start_position + query_base + query_count - 1);
    const int key_tiles = max_query_position >= 0 ?
        (max_query_position + KeyTile) / KeyTile : 0;
    const size_t query_head_offset = static_cast<size_t>(head) * query_head_stride;

    /* 32 KiB Q + 32 KiB K + 16 KiB score + 8 KiB normalized weights +
     * eight private 512-byte V tiles = 92 KiB, below the 99 KiB opt-in
     * limit on Ada/Blackwell. */
    extern __shared__ unsigned char gqa_mma_storage[];
    __half *query_shared = reinterpret_cast<__half *>(gqa_mma_storage);
    __half *key_shared = query_shared + Group * QueryTokens * HeadSize;
    float *score_shared = reinterpret_cast<float *>(key_shared + KeyTile * HeadSize);
    __half *weight_shared = reinterpret_cast<__half *>(score_shared +
        Group * QueryTokens * KeyTile);
    /* Each warp owns one query head.  Keep a private V tile per warp so the
     * value load/MMA phase needs only warp barriers; the old single tile made
     * all four warps overwrite the same addresses and forced a block barrier
     * for every 16-column fragment. */
    __half *value_shared = weight_shared + Group * QueryTokens * KeyTile;
    float running_max = -FLT_MAX;
    float running_norm = 0.0f;
    float tile_rescale = 1.0f;

    for (int index = static_cast<int>(threadIdx.x);
        index < Group * QueryTokens * HeadSize; index += blockDim.x) {
        const int grouped_head = index / (QueryTokens * HeadSize);
        const int remainder = index - grouped_head * QueryTokens * HeadSize;
        const int q = remainder / HeadSize;
        const int component = remainder - q * HeadSize;
        const bool valid = q < query_count;
        const int source_head = kv_head * Group + grouped_head;
        query_shared[index] = valid && source_head < heads ? __float2half(
            query[static_cast<size_t>(query_base + q) * query_stride +
                static_cast<size_t>(source_head) * query_head_stride + component]) :
            __float2half(0.0f);
    }
    __syncthreads();

    for (int key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
        const int key_base = key_tile_index * KeyTile;
        for (int index = static_cast<int>(threadIdx.x);
            index < KeyTile * HeadSize; index += blockDim.x) {
            const int key = index / HeadSize;
            const int component = index - key * HeadSize;
            const int source_key = key_base + key;
                key_shared[index] = source_key < max_context ?
                    qwen_attention_load_cache_component<QuantizedCache, HeadSize>(
                        key_cache, source_key, kv_head, key_value_heads, component) :
                    __float2half(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_accumulator;
        for (int key_group = 0; key_group < KeyTile / key_mma_tile; ++key_group) {
            wmma::fill_fragment(score_accumulator, 0.0f);
            for (int k0 = 0; k0 < HeadSize; k0 += key_mma_tile) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_fragment;
                wmma::load_matrix_sync(q_fragment, query_shared + warp * QueryTokens * HeadSize + k0,
                    HeadSize);
                wmma::load_matrix_sync(k_fragment,
                    key_shared + key_group * key_mma_tile * HeadSize + k0, HeadSize);
                wmma::mma_sync(score_accumulator, q_fragment, k_fragment, score_accumulator);
            }
            wmma::store_matrix_sync(score_shared + warp * QueryTokens * KeyTile +
                key_group * key_mma_tile, score_accumulator, KeyTile, wmma::mem_row_major);
        }
        __syncwarp();

        /* Each lane owns one query row.  The scalar softmax work is tiny
         * compared with the tensor-core dot products and avoids a block-wide
         * reduction or score materialization in global memory. */
        if (lane < QueryTokens) {
            const int q = lane;
            const int position = start_position + query_base + q;
            float tile_maximum = -FLT_MAX;
            for (int key = 0; key < KeyTile; ++key) {
                const int source_key = key_base + key;
                if (q < query_count && source_key <= position && source_key < max_context)
                    tile_maximum = fmaxf(tile_maximum, score_shared[warp * QueryTokens * KeyTile +
                        q * KeyTile + key] / sqrtf(static_cast<float>(HeadSize)));
            }
            const float new_max = fmaxf(running_max, tile_maximum);
            tile_rescale = expf(running_max - new_max);
            float tile_norm = 0.0f;
            for (int key = 0; key < KeyTile; ++key) {
                const int source_key = key_base + key;
                const bool valid = q < query_count && source_key <= position &&
                    source_key < max_context;
                const float weight = valid ? expf(score_shared[warp * QueryTokens * KeyTile +
                    q * KeyTile + key] / sqrtf(static_cast<float>(HeadSize)) - new_max) : 0.0f;
                weight_shared[warp * QueryTokens * KeyTile + q * KeyTile + key] =
                    __float2half(weight);
                tile_norm += weight;
            }
            running_max = new_max;
            running_norm = running_norm * tile_rescale + tile_norm;
        }
        __syncwarp();

        for (int value_base = 0; value_base < ValueSize; value_base += key_mma_tile) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> value_accumulator;
            wmma::fill_fragment(value_accumulator, 0.0f);
            for (int key_group = 0; key_group < KeyTile / key_mma_tile; ++key_group) {
                __half *warp_value_shared = value_shared + warp * key_mma_tile * key_mma_tile;
                for (int index = lane; index < key_mma_tile * key_mma_tile;
                    index += warp_size) {
                    const int value = index / key_mma_tile;
                    const int key = index - value * key_mma_tile;
                    const int source_key = key_base + key_group * key_mma_tile + key;
                    warp_value_shared[index] = source_key < max_context ?
                        qwen_attention_load_cache_component<QuantizedCache, ValueSize>(
                            value_cache, source_key, kv_head, key_value_heads, value_base + value) :
                        __float2half(0.0f);
                }
                __syncwarp();
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> weight_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> value_fragment;
                wmma::load_matrix_sync(weight_fragment,
                    weight_shared + warp * QueryTokens * KeyTile + key_group * key_mma_tile,
                    KeyTile);
                wmma::load_matrix_sync(value_fragment, warp_value_shared, key_mma_tile);
                wmma::mma_sync(value_accumulator, weight_fragment, value_fragment,
                    value_accumulator);
                __syncwarp();
            }
            /* Reuse the score storage for one small output tile.  Lanes 0..15
             * own complete query rows, allowing them to combine the prior
             * tile with the online-softmax rescale without extra metadata or
             * a second global scratch buffer. */
            float *tile_output = score_shared + warp * QueryTokens * key_mma_tile;
            wmma::store_matrix_sync(tile_output, value_accumulator, key_mma_tile,
                wmma::mem_row_major);
            __syncwarp();
            if (lane < QueryTokens && lane < query_count) {
                const size_t output_base = static_cast<size_t>(query_base + lane) *
                    attention_stride + static_cast<size_t>(head) * ValueSize + value_base;
                for (int value = 0; value < key_mma_tile && value_base + value < ValueSize; ++value) {
                    const float previous = key_tile_index == 0 ? 0.0f :
                        attention[output_base + value];
                    attention[output_base + value] = previous * tile_rescale +
                        tile_output[lane * key_mma_tile + value];
                }
            }
            __syncwarp();
        }
        __syncwarp();
    }

    if (lane < QueryTokens && lane < query_count) {
        const float inverse = running_norm > 0.0f ? 1.0f / running_norm : 0.0f;
        const size_t output_base = static_cast<size_t>(query_base + lane) * attention_stride +
            static_cast<size_t>(head) * ValueSize;
        const float *gate = query + static_cast<size_t>(query_base + lane) * query_stride +
            query_head_offset + HeadSize;
        for (int value = 0; value < ValueSize; ++value)
            attention[output_base + value] *= inverse * qwen_sigmoid(gate[value]);
    }
#else
    (void) query; (void) key_cache; (void) value_cache; (void) attention;
    (void) heads; (void) key_value_heads; (void) max_context; (void) start_position;
    (void) query_stride; (void) attention_stride; (void) batch; (void) query_head_stride;
#endif
}

/* Streaming causal softmax for the resident cache.  Unlike the original
 * shared-score implementation this keeps no max_context-sized shared array,
 * so q8/f16 attention remains valid at the model's full context length. */
__global__ void qwen_attention_apply_online(const float *query,
    const __half *key_cache, const int8_t *key_cache_q8,
    const __half *value_cache, const int8_t *value_cache_q8,
    float *attention, int heads, int key_value_heads, int head_size,
    int value_size, int max_context, int cache_key_type, int cache_value_type,
    const int *position_ptr) {
    const int head = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (head >= heads) return;
    const int position = *position_ptr;
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const int query_offset = head * 2 * head_size;
    const int key_row_bytes = cache_key_type == 2 ?
        (head_size / q4_block_width) * q4_block_bytes :
        (head_size / q8_block_width) * q8_block_bytes;
    const int value_row_bytes = cache_value_type == 2 ?
        (value_size / q4_block_width) * q4_block_bytes :
        (value_size / q8_block_width) * q8_block_bytes;
    __shared__ float warp_partials[8];
    __shared__ float shared_score;
    __shared__ float shared_old_max;
    __shared__ float shared_new_max;
    __shared__ float shared_normalizer;
    float running_max = -FLT_MAX;
    float running_normalizer = 0.0f;
    float value_accumulator = 0.0f;

    if (value_size != 256) {
        for (int i = tid; i < value_size; i += blockDim.x)
            attention[head * value_size + i] = 0.0f;
    }
    __syncthreads();

    for (int previous = 0; previous <= position; ++previous) {
        float dot = 0.0f;
        if (cache_key_type == 1) {
            const int8_t *row = key_cache_q8 +
                (static_cast<size_t>(previous) * key_value_heads + kv_head) * key_row_bytes;
            for (int i = tid; i < head_size; i += blockDim.x) {
                const int block = i / q8_block_width;
                const int offset = i % q8_block_width;
                dot += query[query_offset + i] * block_scale(row + block * q8_block_bytes) *
                    static_cast<float>(row[block * q8_block_bytes + 2 + offset]);
            }
        } else if (cache_key_type == 2) {
            const int8_t *row = key_cache_q8 +
                (static_cast<size_t>(previous) * key_value_heads + kv_head) * key_row_bytes;
            for (int i = tid; i < head_size; i += blockDim.x)
                dot += query[query_offset + i] * qwen_load_q4_cache_component(row, i);
        } else {
            const int key_offset = (previous * key_value_heads + kv_head) * head_size;
            for (int i = tid; i < head_size; i += blockDim.x)
                dot += query[query_offset + i] * __half2float(key_cache[key_offset + i]);
        }
        for (int offset = 16; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        if (lane == 0) warp_partials[warp] = dot;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int i = 0; i < (blockDim.x + 31) / 32; ++i) total += warp_partials[i];
            const float score = total / sqrtf(static_cast<float>(head_size));
            const float new_max = fmaxf(running_max, score);
            const float new_normalizer = running_normalizer * expf(running_max - new_max) +
                expf(score - new_max);
            shared_score = score;
            shared_old_max = running_max;
            shared_new_max = new_max;
            running_max = new_max;
            running_normalizer = new_normalizer;
        }
        __syncthreads();
        const float rescale = expf(shared_old_max - shared_new_max);
        const float weight = expf(shared_score - shared_new_max);
        for (int i = tid; i < value_size; i += blockDim.x) {
            float value = 0.0f;
            if (cache_value_type == 1) {
                const int8_t *row = value_cache_q8 +
                    (static_cast<size_t>(previous) * key_value_heads + kv_head) * value_row_bytes;
                const int block = i / q8_block_width;
                const int offset = i % q8_block_width;
                value = block_scale(row + block * q8_block_bytes) *
                    static_cast<float>(row[block * q8_block_bytes + 2 + offset]);
            } else if (cache_value_type == 2) {
                const int8_t *row = value_cache_q8 +
                    (static_cast<size_t>(previous) * key_value_heads + kv_head) * value_row_bytes;
                value = qwen_load_q4_cache_component(row, i);
            } else {
                const int value_offset = (previous * key_value_heads + kv_head) * value_size;
                value = __half2float(value_cache[value_offset + i]);
            }
            const int output_offset = head * value_size + i;
            if (value_size == 256) {
                value_accumulator = value_accumulator * rescale + weight * value;
            } else {
                attention[output_offset] = attention[output_offset] * rescale + weight * value;
            }
        }
    }
    if (tid == 0) shared_normalizer = running_normalizer;
    __syncthreads();
    const float inverse = shared_normalizer > 0.0f ? 1.0f / shared_normalizer : 0.0f;
    const float *gate = query + query_offset + head_size;
    for (int i = tid; i < value_size; i += blockDim.x) {
        const float accumulated = value_size == 256 ? value_accumulator :
            attention[head * value_size + i];
        attention[head * value_size + i] = accumulated * inverse * qwen_sigmoid(gate[i]);
    }
}

__global__ void qwen_attention_prepare_batch(float *query, float *key,
    const float *value, const float *query_norm, const float *key_norm,
    __half *key_cache, __half *value_cache, int8_t *key_cache_q8,
    int8_t *value_cache_q8, int cache_key_type, int cache_value_type, int heads,
    int key_value_heads, int head_size, int value_size, int max_context,
    const int *start_position_ptr, int query_stride, int key_stride, int value_stride,
    int batch, int rope_dimension, float rope_base, float epsilon) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (head >= heads || token >= batch) return;
    const int position = *start_position_ptr + token;
    if (position < 0 || position >= max_context) return;
    extern __shared__ float attention_partial_batch[];
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const bool writes_kv = (head % group) == 0;
    const size_t query_offset = static_cast<size_t>(token) * query_stride +
        static_cast<size_t>(head) * 2 * head_size;
    const size_t key_offset = static_cast<size_t>(token) * key_stride +
        static_cast<size_t>(kv_head) * head_size;
    const size_t value_offset = static_cast<size_t>(token) * value_stride +
        static_cast<size_t>(kv_head) * value_size;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = (blockDim.x + 31) / 32;
    float query_sum = 0.0f;
    for (int i = tid; i < head_size; i += blockDim.x) {
        const float x = query[query_offset + i];
        query_sum += x * x;
    }
    float key_sum = 0.0f;
    if (writes_kv) {
        for (int i = tid; i < head_size; i += blockDim.x) {
            const float x = key[key_offset + i];
            key_sum += x * x;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        query_sum += __shfl_xor_sync(0xffffffffu, query_sum, offset);
        key_sum += __shfl_xor_sync(0xffffffffu, key_sum, offset);
    }
    if (lane == 0) {
        attention_partial_batch[warp] = query_sum;
        attention_partial_batch[blockDim.x + warp] = key_sum;
    }
    __syncthreads();
    if (warp == 0) {
        query_sum = lane < warps ? attention_partial_batch[lane] : 0.0f;
        key_sum = lane < warps ? attention_partial_batch[blockDim.x + lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            query_sum += __shfl_xor_sync(0xffffffffu, query_sum, offset);
            key_sum += __shfl_xor_sync(0xffffffffu, key_sum, offset);
        }
        if (lane == 0) {
            attention_partial_batch[0] = query_sum;
            attention_partial_batch[blockDim.x] = key_sum;
        }
    }
    __syncthreads();
    const float query_inverse = rsqrtf(attention_partial_batch[0] /
        static_cast<float>(head_size) + epsilon);
    for (int i = tid; i < head_size; i += blockDim.x)
        query[query_offset + i] *= query_inverse * query_norm[i];
    __syncthreads();
    if (rope_dimension > 0) {
        for (int i = tid; i < rope_dimension / 2; i += blockDim.x)
            qwen_rope_pair(query + query_offset, rope_dimension, i, position, rope_base);
    }
    if (!writes_kv) return;
    const float key_inverse = rsqrtf(attention_partial_batch[blockDim.x] /
        static_cast<float>(head_size) + epsilon);
    for (int i = tid; i < head_size; i += blockDim.x)
        key[key_offset + i] *= key_inverse * key_norm[i];
    __syncthreads();
    if (rope_dimension > 0) {
        for (int i = tid; i < rope_dimension / 2; i += blockDim.x)
            qwen_rope_pair(key + key_offset, rope_dimension, i, position, rope_base);
    }
    __syncthreads();
    if (cache_key_type == 1) {
        const int blocks = head_size / q8_block_width;
        const size_t row_offset = (static_cast<size_t>(position) * key_value_heads + kv_head) *
            static_cast<size_t>(blocks) * q8_block_bytes;
        for (int block = tid; block < blocks; block += blockDim.x)
            qwen_quantize_cache_block(key + key_offset, key_cache_q8 + row_offset, block);
    } else if (cache_key_type == 2) {
        const int blocks = head_size / q4_block_width;
        const size_t row_offset = (static_cast<size_t>(position) * key_value_heads + kv_head) *
            static_cast<size_t>(blocks) * q4_block_bytes;
        for (int block = tid; block < blocks; block += blockDim.x)
            qwen_quantize_cache_block_q4(key + key_offset, key_cache_q8 + row_offset, block);
    } else {
        for (int i = tid; i < head_size; i += blockDim.x)
            key_cache[(static_cast<size_t>(position) * key_value_heads + kv_head) * head_size + i] =
                __float2half_rn(key[key_offset + i]);
    }
    if (cache_value_type == 1) {
        const int blocks = value_size / q8_block_width;
        const size_t row_offset = (static_cast<size_t>(position) * key_value_heads + kv_head) *
            static_cast<size_t>(blocks) * q8_block_bytes;
        for (int block = tid; block < blocks; block += blockDim.x)
            qwen_quantize_cache_block(value + value_offset, value_cache_q8 + row_offset, block);
    } else if (cache_value_type == 2) {
        const int blocks = value_size / q4_block_width;
        const size_t row_offset = (static_cast<size_t>(position) * key_value_heads + kv_head) *
            static_cast<size_t>(blocks) * q4_block_bytes;
        for (int block = tid; block < blocks; block += blockDim.x)
            qwen_quantize_cache_block_q4(value + value_offset, value_cache_q8 + row_offset, block);
    } else {
        for (int i = tid; i < value_size; i += blockDim.x)
            value_cache[(static_cast<size_t>(position) * key_value_heads + kv_head) * value_size + i] =
                __float2half_rn(value[value_offset + i]);
    }
}

__global__ void qwen_attention_apply_online_batch(const float *query,
    const __half *key_cache, const int8_t *key_cache_q8,
    const __half *value_cache, const int8_t *value_cache_q8,
    float *attention, int heads, int key_value_heads, int head_size,
    int value_size, int max_context, int cache_key_type, int cache_value_type,
    const int *start_position_ptr, int query_stride, int attention_stride, int batch,
    float *partial_output = nullptr, int max_partitions = 1) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (head >= heads || token >= batch) return;
    const int position = *start_position_ptr + token;
    if (position < 0 || position >= max_context) return;
    const int partition_count = max_partitions > 1 && position >= 512 ?
        max_partitions : 1;
    const int partition = static_cast<int>(blockIdx.z);
    if (partition >= partition_count) return;
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const size_t query_offset = static_cast<size_t>(token) * query_stride +
        static_cast<size_t>(head) * 2 * head_size;
    const size_t output_offset = static_cast<size_t>(token) * attention_stride +
        static_cast<size_t>(head) * value_size;
    const int key_row_bytes = cache_key_type == 2 ?
        (head_size / q4_block_width) * q4_block_bytes :
        (head_size / q8_block_width) * q8_block_bytes;
    const int value_row_bytes = cache_value_type == 2 ?
        (value_size / q4_block_width) * q4_block_bytes :
        (value_size / q8_block_width) * q8_block_bytes;
    __shared__ float warp_partials[8];
    __shared__ float shared_score;
    __shared__ float shared_old_max;
    __shared__ float shared_new_max;
    __shared__ float shared_normalizer;
    float running_max = -FLT_MAX;
    float running_normalizer = 0.0f;
    float value_accumulator = 0.0f;
    if (value_size != 256) {
        for (int i = tid; i < value_size; i += blockDim.x)
            attention[output_offset + i] = 0.0f;
    }
    __syncthreads();
    const int key_count = position + 1;
    const int first_key = key_count * partition / partition_count;
    const int last_key = key_count * (partition + 1) / partition_count;
    for (int previous = first_key; previous < last_key; ++previous) {
        float dot = 0.0f;
        if (cache_key_type == 1) {
            const int8_t *row = key_cache_q8 +
                (static_cast<size_t>(previous) * key_value_heads + kv_head) * key_row_bytes;
            for (int i = tid; i < head_size; i += blockDim.x) {
                const int block = i / q8_block_width;
                const int offset = i % q8_block_width;
                dot += query[query_offset + i] * block_scale(row + block * q8_block_bytes) *
                    static_cast<float>(row[block * q8_block_bytes + 2 + offset]);
            }
        } else if (cache_key_type == 2) {
            const int8_t *row = key_cache_q8 +
                (static_cast<size_t>(previous) * key_value_heads + kv_head) * key_row_bytes;
            for (int i = tid; i < head_size; i += blockDim.x)
                dot += query[query_offset + i] * qwen_load_q4_cache_component(row, i);
        } else {
            const int key_offset = (previous * key_value_heads + kv_head) * head_size;
            for (int i = tid; i < head_size; i += blockDim.x)
                dot += query[query_offset + i] * __half2float(key_cache[key_offset + i]);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        if (lane == 0) warp_partials[warp] = dot;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int i = 0; i < (blockDim.x + 31) / 32; ++i) total += warp_partials[i];
            const float score = total / sqrtf(static_cast<float>(head_size));
            const float new_max = fmaxf(running_max, score);
            const float new_normalizer = running_normalizer * expf(running_max - new_max) +
                expf(score - new_max);
            shared_score = score;
            shared_old_max = running_max;
            shared_new_max = new_max;
            running_max = new_max;
            running_normalizer = new_normalizer;
        }
        __syncthreads();
        const float rescale = expf(shared_old_max - shared_new_max);
        const float weight = expf(shared_score - shared_new_max);
        for (int i = tid; i < value_size; i += blockDim.x) {
            float value = 0.0f;
            if (cache_value_type == 1) {
                const int8_t *row = value_cache_q8 +
                    (static_cast<size_t>(previous) * key_value_heads + kv_head) * value_row_bytes;
                const int block = i / q8_block_width;
                const int offset = i % q8_block_width;
                value = block_scale(row + block * q8_block_bytes) *
                    static_cast<float>(row[block * q8_block_bytes + 2 + offset]);
            } else if (cache_value_type == 2) {
                const int8_t *row = value_cache_q8 +
                    (static_cast<size_t>(previous) * key_value_heads + kv_head) * value_row_bytes;
                value = qwen_load_q4_cache_component(row, i);
            } else {
                const int value_offset = (previous * key_value_heads + kv_head) * value_size;
                value = __half2float(value_cache[value_offset + i]);
            }
            if (value_size == 256) {
                value_accumulator = value_accumulator * rescale + weight * value;
            } else {
                attention[output_offset + i] = attention[output_offset + i] * rescale + weight * value;
            }
        }
    }
    if (partial_output && value_size == 256) {
        const size_t partial_offset =
            ((static_cast<size_t>(token) * heads + head) * max_partitions +
                partition) * (value_size + 2);
        if (tid == 0) {
            partial_output[partial_offset] = running_max;
            partial_output[partial_offset + 1] = running_normalizer;
        }
        partial_output[partial_offset + 2 + tid] = value_accumulator;
    } else {
        if (tid == 0) shared_normalizer = running_normalizer;
        __syncthreads();
        const float inverse = shared_normalizer > 0.0f ?
            1.0f / shared_normalizer : 0.0f;
        const float *gate = query + query_offset + head_size;
        for (int i = tid; i < value_size; i += blockDim.x) {
            const float accumulated = value_size == 256 ? value_accumulator :
                attention[output_offset + i];
            attention[output_offset + i] = accumulated * inverse * qwen_sigmoid(gate[i]);
        }
    }
}

/* Q4 decode counterpart of the grouped Q8 vector path above.  One block owns
 * a KV head and all of its GQA query heads, so each packed K/V tile is fetched
 * once instead of Group times.  A warp retains one query head's online
 * softmax and eight value components per lane. */
template <int Group, int QueryTokens = 1, int KeyTile = 32>
__launch_bounds__(Group * 32, 1)
__global__ void qwen_attention_apply_gqa_q4_batch(
    const float *__restrict__ query,
    const int8_t *__restrict__ key_cache_q4,
    const int8_t *__restrict__ value_cache_q4,
    float *__restrict__ attention,
    int heads, int key_value_heads, int max_context,
    const int *start_position_ptr, int query_stride,
    int attention_stride, int batch, float *__restrict__ partial_output,
    int max_partitions) {
    constexpr int HeadSize = 256;
    constexpr int ValueSize = 256;
    constexpr int key_row_bytes = (HeadSize / q4_block_width) * q4_block_bytes;
    constexpr int value_row_bytes = (ValueSize / q4_block_width) * q4_block_bytes;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int query_base = static_cast<int>(blockIdx.y) * QueryTokens;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (kv_head >= key_value_heads || Group * key_value_heads != heads ||
        query_base >= batch || warp >= Group) return;

    const int query_count = min(QueryTokens, batch - query_base);
    const int position_base = *start_position_ptr;
    const int max_query_position = min(max_context - 1,
        position_base + query_base + query_count - 1);
    const int partition_count = max_partitions > 1 && max_query_position >= 512 ?
        max_partitions : 1;
    const int partition = static_cast<int>(blockIdx.z);
    if (partition >= partition_count) return;

    extern __shared__ unsigned char storage[];
    auto *key_tile = reinterpret_cast<int8_t *>(storage);
    auto *value_tile = key_tile + KeyTile * key_row_bytes;
    auto *scores = reinterpret_cast<float *>(value_tile + KeyTile * value_row_bytes);
    const int query_head = kv_head * Group + warp;
    const int query_head_offset = query_head * 2 * HeadSize;
    const int component_base = lane * (ValueSize / 32);

    float accumulator[QueryTokens][ValueSize / 32];
    float running_max[QueryTokens];
    float running_norm[QueryTokens];
#pragma unroll
    for (int q = 0; q < QueryTokens; ++q) {
        running_max[q] = -FLT_MAX / 2.0f;
        running_norm[q] = 0.0f;
#pragma unroll
        for (int i = 0; i < ValueSize / 32; ++i) accumulator[q][i] = 0.0f;
    }

    const int key_tile_count = max_query_position >= 0 ?
        (max_query_position + KeyTile) / KeyTile : 0;
    const int first_key_tile = key_tile_count * partition / partition_count;
    const int last_key_tile = key_tile_count * (partition + 1) / partition_count;
    for (int tile_index = first_key_tile; tile_index < last_key_tile; ++tile_index) {
        const int key_base = tile_index * KeyTile;
        /* Stage K and V with 16-byte vector copies.  A byte-at-a-time loop
         * issues one memory transaction per element; at 16k context this
         * kernel moved only 12% of peak bandwidth while accounting for 82% of
         * decode GPU time.  llama.cpp's fattn kernels copy K/V in 16-byte
         * units for the same reason (ggml_cuda_memcpy_1<cpy_nb>). */
        constexpr int copy_bytes = static_cast<int>(sizeof(int4));
        static_assert(key_row_bytes % copy_bytes == 0, "key row must be 16-byte divisible");
        static_assert(value_row_bytes % copy_bytes == 0, "value row must be 16-byte divisible");
        constexpr int key_chunks = key_row_bytes / copy_bytes;
        constexpr int value_chunks = value_row_bytes / copy_bytes;
        for (int chunk = tid; chunk < KeyTile * key_chunks; chunk += blockDim.x) {
            const int key = chunk / key_chunks;
            const int source = key_base + key;
            int4 packed = make_int4(0, 0, 0, 0);
            if (source <= max_query_position) {
                const int4 *row = reinterpret_cast<const int4 *>(key_cache_q4 +
                    (static_cast<size_t>(source) * key_value_heads + kv_head) *
                        static_cast<size_t>(key_row_bytes));
                packed = row[chunk - key * key_chunks];
            }
            reinterpret_cast<int4 *>(key_tile)[chunk] = packed;
        }
        for (int chunk = tid; chunk < KeyTile * value_chunks; chunk += blockDim.x) {
            const int key = chunk / value_chunks;
            const int source = key_base + key;
            int4 packed = make_int4(0, 0, 0, 0);
            if (source <= max_query_position) {
                const int4 *row = reinterpret_cast<const int4 *>(value_cache_q4 +
                    (static_cast<size_t>(source) * key_value_heads + kv_head) *
                        static_cast<size_t>(value_row_bytes));
                packed = row[chunk - key * value_chunks];
            }
            reinterpret_cast<int4 *>(value_tile)[chunk] = packed;
        }
        __syncthreads();

        for (int local_key = 0; local_key < KeyTile; ++local_key) {
            const int source = key_base + local_key;
            const int8_t *key_row = key_tile + local_key * key_row_bytes;
#pragma unroll
            for (int q = 0; q < QueryTokens; ++q) {
                const size_t query_offset = static_cast<size_t>(query_base + q) * query_stride +
                    query_head_offset;
                float dot = 0.0f;
#pragma unroll
                for (int i = lane; i < HeadSize; i += 32)
                    dot += query[query_offset + i] * qwen_load_q4_cache_component(key_row, i);
                for (int offset = 16; offset > 0; offset >>= 1)
                    dot += __shfl_down_sync(0xffffffffu, dot, offset);
                if (lane == 0)
                    scores[(warp * QueryTokens + q) * KeyTile + local_key] =
                        source <= max_query_position ? dot * (1.0f / 16.0f) : -FLT_MAX / 2.0f;
            }
        }
        __syncwarp();

        float tile_new_max[QueryTokens];
        float tile_rescale[QueryTokens];
#pragma unroll
        for (int q = 0; q < QueryTokens; ++q) {
            const int position = position_base + query_base + q;
            const int source = key_base + lane;
            const bool valid = q < query_count && source <= position &&
                source <= max_query_position;
            const float score = scores[(warp * QueryTokens + q) * KeyTile + lane];
            float tile_max = valid ? score : -FLT_MAX / 2.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                tile_max = fmaxf(tile_max,
                    __shfl_down_sync(0xffffffffu, tile_max, offset));
            tile_max = __shfl_sync(0xffffffffu, tile_max, 0);
            tile_new_max[q] = fmaxf(running_max[q], tile_max);
            tile_rescale[q] = expf(running_max[q] - tile_new_max[q]);
            float tile_norm = valid ? expf(score - tile_new_max[q]) : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                tile_norm += __shfl_down_sync(0xffffffffu, tile_norm, offset);
            tile_norm = __shfl_sync(0xffffffffu, tile_norm, 0);
            running_norm[q] = running_norm[q] * tile_rescale[q] + tile_norm;
#pragma unroll
            for (int i = 0; i < ValueSize / 32; ++i)
                accumulator[q][i] *= tile_rescale[q];
        }

        for (int local_key = 0; local_key < KeyTile; ++local_key) {
            const int source = key_base + local_key;
            if (source > max_query_position) continue;
            const int8_t *value_row = value_tile + local_key * value_row_bytes;
#pragma unroll
            for (int q = 0; q < QueryTokens; ++q) {
                const int position = position_base + query_base + q;
                if (q >= query_count || source > position) continue;
                const float weight = expf(
                    scores[(warp * QueryTokens + q) * KeyTile + local_key] - tile_new_max[q]);
#pragma unroll
                for (int i = 0; i < ValueSize / 32; ++i)
                    accumulator[q][i] += weight *
                        qwen_load_q4_cache_component(value_row, component_base + i);
            }
        }
#pragma unroll
        for (int q = 0; q < QueryTokens; ++q) running_max[q] = tile_new_max[q];
        __syncthreads();
    }

#pragma unroll
    for (int q = 0; q < QueryTokens; ++q) {
        if (q >= query_count) continue;
        /* Unpartitioned launches own the whole key range, so this block already
         * holds the final softmax state.  Normalize and gate here rather than
         * staging through the partition scratch: that buffer is grown with
         * cudaMalloc, which cannot run inside a CUDA graph capture, and the
         * combine pass would only re-read what is already in registers. */
        if (partial_output == nullptr) {
            const size_t query_offset = static_cast<size_t>(query_base + q) * query_stride +
                query_head_offset;
            const float inverse = running_norm[q] > 0.0f ? 1.0f / running_norm[q] : 0.0f;
#pragma unroll
            for (int i = 0; i < ValueSize / 32; ++i) {
                const int component = component_base + i;
                const float gate = query[query_offset + ValueSize + component];
                attention[static_cast<size_t>(query_base + q) * attention_stride +
                    static_cast<size_t>(query_head) * ValueSize + component] =
                    accumulator[q][i] * inverse * qwen_sigmoid(gate);
            }
            continue;
        }
        const size_t row = static_cast<size_t>(query_base + q) * heads + query_head;
        const size_t partial_offset =
            (row * max_partitions + partition) * (ValueSize + 2);
        if (lane == 0) {
            partial_output[partial_offset] = running_max[q];
            partial_output[partial_offset + 1] = running_norm[q];
        }
#pragma unroll
        for (int i = 0; i < ValueSize / 32; ++i)
            partial_output[partial_offset + 2 + component_base + i] = accumulator[q][i];
    }
}

struct fortai_cuda_qwen35_attention_impl {
    fortai_cuda_q8_context_impl *context = nullptr;
    fortai_cuda_q8_weights_impl *query_weights = nullptr;
    fortai_cuda_q8_weights_impl *key_weights = nullptr;
    fortai_cuda_q8_weights_impl *value_weights = nullptr;
    fortai_cuda_q8_weights_impl *output_weights = nullptr;
    float *query_norm = nullptr;
    float *key_norm = nullptr;
    __half *key_cache = nullptr;
    __half *value_cache = nullptr;
    int8_t *key_cache_q8 = nullptr;
    int8_t *value_cache_q8 = nullptr;
    float *query = nullptr;
    float *key = nullptr;
    float *value = nullptr;
    float *attention = nullptr;
    int heads = 0;
    int key_value_heads = 0;
    int head_size = 0;
    int value_size = 0;
    int max_context = 0;
    int rope_dimension = 0;
    float rope_base = 1.0f;
    float norm_epsilon = 1.0e-6f;
    /* 0 = F16, 1 = Q8_0, 2 = Q4_0. */
    int cache_key_type = 0;
    int cache_value_type = 0;
};

static size_t attention_cache_row_bytes(int cache_type, int width) {
    if (cache_type == 1)
        return static_cast<size_t>(width / q8_block_width) * q8_block_bytes;
    if (cache_type == 2)
        return static_cast<size_t>(width / q4_block_width) * q4_block_bytes;
    return static_cast<size_t>(width) * sizeof(__half);
}

struct fortai_cuda_qwen35_recurrent_impl {
    fortai_cuda_q8_context_impl *context = nullptr;
    fortai_cuda_q8_weights_impl *qkv_weights = nullptr;
    fortai_cuda_q8_weights_impl *gate_weights = nullptr;
    fortai_cuda_q8_weights_impl *alpha_weights = nullptr;
    fortai_cuda_q8_weights_impl *beta_weights = nullptr;
    fortai_cuda_q8_weights_impl *output_weights = nullptr;
    float *conv_weights = nullptr;
    float *ssm_a = nullptr;
    float *ssm_dt = nullptr;
    float *ssm_norm = nullptr;
    float *conv_state = nullptr;
    float *gdn_state = nullptr;
    float *conv_state_first = nullptr;
    float *gdn_state_first = nullptr;
    float *conv_state_second = nullptr;
    float *gdn_state_second = nullptr;
    float *qkv_output = nullptr;
    float *gate_output = nullptr;
    float *alpha_output = nullptr;
    float *beta_output = nullptr;
    float *gdn_output = nullptr;
    int conv_size = 0;
    int conv_kernel = 0;
    int state_size = 0;
    int key_heads = 0;
    int value_heads = 0;
    int head_size = 0;
    int inner_size = 0;
    float norm_epsilon = 1.0e-6f;
};

static void free_recurrent_device_buffers(fortai_cuda_qwen35_recurrent_impl *layer) {
    if (!layer) return;
    cudaFree(layer->conv_weights);
    cudaFree(layer->ssm_a);
    cudaFree(layer->ssm_dt);
    cudaFree(layer->ssm_norm);
    cudaFree(layer->conv_state);
    cudaFree(layer->gdn_state);
    cudaFree(layer->conv_state_first);
    cudaFree(layer->gdn_state_first);
    cudaFree(layer->conv_state_second);
    cudaFree(layer->gdn_state_second);
    cudaFree(layer->qkv_output);
    cudaFree(layer->gate_output);
    cudaFree(layer->alpha_output);
    cudaFree(layer->beta_output);
    cudaFree(layer->gdn_output);
    layer->conv_weights = nullptr;
    layer->ssm_a = nullptr;
    layer->ssm_dt = nullptr;
    layer->ssm_norm = nullptr;
    layer->conv_state = nullptr;
    layer->gdn_state = nullptr;
    layer->conv_state_first = nullptr;
    layer->gdn_state_first = nullptr;
    layer->conv_state_second = nullptr;
    layer->gdn_state_second = nullptr;
    layer->qkv_output = nullptr;
    layer->gate_output = nullptr;
    layer->alpha_output = nullptr;
    layer->beta_output = nullptr;
    layer->gdn_output = nullptr;
}

static cudaError_t allocate_and_copy_float(fortai_cuda_qwen35_recurrent_impl *layer,
    float **destination, const void *source, size_t bytes) {
    cudaError_t error = cudaMalloc(reinterpret_cast<void **>(destination), bytes);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(*destination, source, bytes, cudaMemcpyHostToDevice,
            layer->context->stream);
    return error;
}

static void free_attention_device_buffers(fortai_cuda_qwen35_attention_impl *layer) {
    if (!layer) return;
    cudaFree(layer->query_norm);
    cudaFree(layer->key_norm);
    cudaFree(layer->key_cache);
    cudaFree(layer->value_cache);
    cudaFree(layer->key_cache_q8);
    cudaFree(layer->value_cache_q8);
    cudaFree(layer->query);
    cudaFree(layer->key);
    cudaFree(layer->value);
    cudaFree(layer->attention);
    layer->query_norm = nullptr;
    layer->key_norm = nullptr;
    layer->key_cache = nullptr;
    layer->value_cache = nullptr;
    layer->key_cache_q8 = nullptr;
    layer->value_cache_q8 = nullptr;
    layer->query = nullptr;
    layer->key = nullptr;
    layer->value = nullptr;
    layer->attention = nullptr;
}

static cudaError_t allocate_and_copy_float_attention(fortai_cuda_qwen35_attention_impl *layer,
    float **destination, const void *source, size_t bytes) {
    cudaError_t error = cudaMalloc(reinterpret_cast<void **>(destination), bytes);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(*destination, source, bytes, cudaMemcpyHostToDevice,
            layer->context->stream);
    return error;
}

} // namespace

struct fortai_cuda_q8_context {
    fortai_cuda_q8_context_impl impl;
};

struct fortai_cuda_q8_weights {
    fortai_cuda_q8_weights_impl impl;
};

struct fortai_cuda_qwen35_recurrent {
    fortai_cuda_qwen35_recurrent_impl impl;
};

struct fortai_cuda_qwen35_attention {
    fortai_cuda_qwen35_attention_impl impl;
};

extern "C" int fortai_cuda_q8_context_create(int device,
    fortai_cuda_q8_context **context) {
    if (!context || device < 0) return FORTAI_CUDA_INVALID;
    *context = nullptr;
    auto *created = new (std::nothrow) fortai_cuda_q8_context;
    if (!created) return FORTAI_CUDA_RUNTIME_ERROR;
    created->impl.device = device;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        delete created;
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    const char *disable_mma = std::getenv("FORTAI_CUDA_MMA");
    created->impl.mma_available = properties.major >= 8 &&
        !(disable_mma && disable_mma[0] == '0');
    created->impl.multiprocessor_count = properties.multiProcessorCount;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaStreamCreateWithFlags(&created->impl.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreate(&created->impl.start) != cudaSuccess ||
        cudaEventCreate(&created->impl.stop) != cudaSuccess) {
        cudaStreamDestroy(created->impl.stream);
        cudaEventDestroy(created->impl.start);
        cudaEventDestroy(created->impl.stop);
        delete created;
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&created->impl.position), sizeof(int)) != cudaSuccess ||
        cudaMemsetAsync(created->impl.position, 0, sizeof(int), created->impl.stream) != cudaSuccess ||
        cudaStreamSynchronize(created->impl.stream) != cudaSuccess) {
        cudaFree(created->impl.position);
        cudaEventDestroy(created->impl.start);
        cudaEventDestroy(created->impl.stop);
        cudaStreamDestroy(created->impl.stream);
        delete created;
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    *context = created;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_memory_info(int device, size_t *free_bytes,
    size_t *total_bytes) {
    if (device < 0 || !free_bytes || !total_bytes) return FORTAI_CUDA_INVALID;
    /* The model runner probes the optional second board before creating its
     * primary context.  An absent ordinal must not leave a sticky CUDA error
     * for the first subsequent kernel launch on a valid device. */
    const cudaError_t set_error = cudaSetDevice(device);
    if (set_error != cudaSuccess) {
        (void) cudaGetLastError();
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    const cudaError_t error = cudaMemGetInfo(free_bytes, total_bytes);
    return error == cudaSuccess ? FORTAI_CUDA_OK : FORTAI_CUDA_RUNTIME_ERROR;
}

extern "C" int fortai_cuda_q8_context_destroy(fortai_cuda_q8_context *context) {
    if (!context) return FORTAI_CUDA_OK;
    cudaSetDevice(context->impl.device);
    /* CUDA rejects destruction of a null graph handle.  Most model runs do
     * not capture a graph, so keep teardown quiet and sanitizer-clean. */
    for (int slot = 0; slot < qwen_graph_slots; ++slot) {
        if (context->impl.graph_execs[slot]) {
            cudaGraphExecDestroy(context->impl.graph_execs[slot]);
            context->impl.graph_execs[slot] = nullptr;
        }
        if (context->impl.graphs[slot]) {
            cudaGraphDestroy(context->impl.graphs[slot]);
            context->impl.graphs[slot] = nullptr;
        }
    }
    cudaFree(context->impl.position);
    cudaFree(context->impl.scratch_activation);
    cudaFree(context->impl.scratch_raw);
    cudaFree(context->impl.scratch_output);
    cudaFree(context->impl.scratch_aux);
    cudaFree(context->impl.scratch_batched_activation);
    cudaFree(context->impl.scratch_mmq_fixup);
    cudaFree(context->impl.scratch_mmq_tiles);
    cudaFree(context->impl.scratch_tokens);
    cudaFree(context->impl.scratch_attention_key_f16);
    cudaFree(context->impl.scratch_attention_value_f16);
    cudaFree(context->impl.scratch_attention_mask_f16);
    cudaFree(context->impl.scratch_attention_meta);
    cudaFreeHost(context->impl.download_host);
    for (int row = 0; row < 32; ++row) {
        if (context->impl.topk_done[row]) cudaEventDestroy(context->impl.topk_done[row]);
        if (context->impl.topk_streams[row]) cudaStreamDestroy(context->impl.topk_streams[row]);
    }
    if (context->impl.topk_ready) cudaEventDestroy(context->impl.topk_ready);
    cudaEventDestroy(context->impl.start);
    cudaEventDestroy(context->impl.stop);
    if (context->impl.owns_stream && context->impl.stream != nullptr)
        cudaStreamDestroy(context->impl.stream);
    context->impl.stream = nullptr;
    delete context;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_q8_context_set_position(fortai_cuda_q8_context *context, int position) {
    if (!context || !context->impl.position || position < 0) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaMemcpyAsync(context->impl.position, &position, sizeof(position),
        cudaMemcpyHostToDevice, context->impl.stream);
    if (error == cudaSuccess) context->impl.position_value = position;
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "set device position", error);
}

extern "C" int fortai_cuda_q8_context_synchronize(fortai_cuda_q8_context *context) {
    if (!context) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaStreamSynchronize(context->impl.stream);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "CUDA context synchronize", error);
}

extern "C" void *fortai_cuda_q8_context_stream(fortai_cuda_q8_context *context) {
    if (!context) return nullptr;
    return reinterpret_cast<void *>(context->impl.stream);
}

extern "C" int fortai_cuda_q8_context_adopt_stream(fortai_cuda_q8_context *context,
    void *stream) {
    if (context == nullptr || stream == nullptr) return FORTAI_CUDA_INVALID;
    cudaStream_t borrowed = reinterpret_cast<cudaStream_t>(stream);
    if (borrowed == context->impl.stream) return FORTAI_CUDA_OK;
    if (cudaSetDevice(context->impl.device) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    /* Uploads and any scratch work queued before the hand-off must complete
     * before the old stream is released.  The setup path performs this once,
     * outside the decode loop. */
    if (context->impl.stream != nullptr &&
        cudaStreamSynchronize(context->impl.stream) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    if (context->impl.owns_stream && context->impl.stream != nullptr &&
        cudaStreamDestroy(context->impl.stream) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    context->impl.stream = borrowed;
    context->impl.owns_stream = false;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_q8_context_capture_begin(fortai_cuda_q8_context *context) {
    return fortai_cuda_q8_context_capture_begin_slot(context, 0);
}

extern "C" int fortai_cuda_q8_context_capture_begin_slot(fortai_cuda_q8_context *context,
    int slot) {
    if (!context || slot < 0 || slot >= qwen_graph_slots ||
        context->impl.graphs[slot] || context->impl.graph_execs[slot])
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaStreamBeginCapture(context->impl.stream,
        cudaStreamCaptureModeGlobal);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "begin CUDA graph capture", error);
}

extern "C" int fortai_cuda_q8_context_capture_end(fortai_cuda_q8_context *context) {
    return fortai_cuda_q8_context_capture_end_slot(context, 0);
}

extern "C" int fortai_cuda_q8_context_capture_end_slot(fortai_cuda_q8_context *context,
    int slot) {
    if (!context || slot < 0 || slot >= qwen_graph_slots) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    cudaError_t error = cudaStreamEndCapture(context->impl.stream, &context->impl.graphs[slot]);
    if (error == cudaSuccess)
        error = cudaGraphInstantiate(&context->impl.graph_execs[slot], context->impl.graphs[slot],
            nullptr, nullptr, 0);
    if (error != cudaSuccess) {
        if (context->impl.graph_execs[slot])
            cudaGraphExecDestroy(context->impl.graph_execs[slot]);
        if (context->impl.graphs[slot])
            cudaGraphDestroy(context->impl.graphs[slot]);
        context->impl.graph_execs[slot] = nullptr;
        context->impl.graphs[slot] = nullptr;
        return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "end CUDA graph capture", error);
    }
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_q8_context_graph_launch(fortai_cuda_q8_context *context) {
    return fortai_cuda_q8_context_graph_launch_slot(context, 0);
}

extern "C" int fortai_cuda_q8_context_graph_launch_slot(fortai_cuda_q8_context *context,
    int slot) {
    if (!context || slot < 0 || slot >= qwen_graph_slots ||
        !context->impl.graph_execs[slot]) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaGraphLaunch(context->impl.graph_execs[slot], context->impl.stream);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "launch CUDA graph", error);
}

extern "C" int fortai_cuda_q8_weights_upload(fortai_cuda_q8_context *context,
    const void *host_weights, size_t weight_bytes, int rows, int width,
    fortai_cuda_q8_weights **weights) {
    if (!context || !host_weights || !weights || rows <= 0 || width <= 0 ||
        width % q8_block_width != 0) return FORTAI_CUDA_INVALID;
    *weights = nullptr;
    const int blocks = width / q8_block_width;
    const size_t expected = static_cast<size_t>(rows) * blocks * q8_block_bytes;
    if (weight_bytes < expected) return FORTAI_CUDA_INVALID;
    auto *created = new (std::nothrow) fortai_cuda_q8_weights;
    if (!created) return FORTAI_CUDA_RUNTIME_ERROR;
    created->impl.context = &context->impl;
    created->impl.rows = rows;
    created->impl.width = width;
    created->impl.blocks = blocks;
    cudaSetDevice(context->impl.device);
    cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&created->impl.device_data), expected);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(created->impl.device_data, host_weights, expected,
            cudaMemcpyHostToDevice, context->impl.stream);
    /* Uploads are queued on the context stream.  Callers submit the complete
     * weight set before the one-time context fence in the model setup path;
     * keeping that fence out of this per-tensor routine removes an O(tensor
     * count) host round-trip from large GGUF loads while preserving stream
     * ordering for every subsequent matvec. */
    if (error != cudaSuccess) {
        cudaFree(created->impl.device_data);
        delete created;
        return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "weight upload", error);
    }
    *weights = created;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_q8_weights_destroy(fortai_cuda_q8_weights *weights) {
    if (!weights) return FORTAI_CUDA_OK;
    if (weights->impl.context) cudaSetDevice(weights->impl.context->device);
    cudaFree(weights->impl.device_data);
    delete weights;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_q8_device_buffer_create(fortai_cuda_q8_context *context,
    size_t bytes, void **device_buffer) {
    if (!context || !device_buffer || bytes == 0) return FORTAI_CUDA_INVALID;
    *device_buffer = nullptr;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaMalloc(device_buffer, bytes);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device buffer allocation", error);
}

extern "C" int fortai_cuda_q8_device_buffer_destroy(fortai_cuda_q8_context *context,
    void *device_buffer) {
    if (!context || !device_buffer) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaFree(device_buffer);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device buffer free", error);
}

extern "C" int fortai_cuda_q8_device_buffer_upload(fortai_cuda_q8_context *context,
    void *device_buffer, const void *host_data, size_t bytes) {
    if (!context || !device_buffer || !host_data || bytes == 0) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaMemcpyAsync(device_buffer, host_data, bytes,
        cudaMemcpyHostToDevice, context->impl.stream);
    if (error == cudaSuccess) {
        const cudaError_t sync = cudaStreamSynchronize(context->impl.stream);
        if (sync != cudaSuccess)
            return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device upload", sync);
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device upload", error);
}

extern "C" int fortai_cuda_q8_device_buffer_upload_ptr(fortai_cuda_q8_context *context,
    void *device_buffer, const void *host_data, size_t bytes) {
    if (!context || !device_buffer || !host_data || bytes == 0) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaMemcpyAsync(device_buffer, host_data, bytes,
        cudaMemcpyHostToDevice, context->impl.stream);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "async device upload", error);
}

extern "C" int fortai_cuda_q8_device_buffer_download(fortai_cuda_q8_context *context,
    void *host_data, const void *device_buffer, size_t bytes) {
    if (!context || !device_buffer || !host_data || bytes == 0) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    cudaError_t error = ensure_download_host(&context->impl, bytes);
    if (error == cudaSuccess) error = cudaMemcpyAsync(context->impl.download_host, device_buffer, bytes,
        cudaMemcpyDeviceToHost, context->impl.stream);
    if (error == cudaSuccess) {
        const cudaError_t sync = cudaStreamSynchronize(context->impl.stream);
        if (sync != cudaSuccess)
            return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device download", sync);
        std::memcpy(host_data, context->impl.download_host, bytes);
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device download", error);
}

extern "C" int fortai_cuda_q8_matvec_resident(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *device_activation,
    void *device_output, float *kernel_ms) {
    if (!context || !weights || weights->impl.context != &context->impl ||
        !device_activation || !device_output || !kernel_ms) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const size_t activation_bytes = static_cast<size_t>(weights->impl.blocks) * q8_block_bytes;
    cudaError_t error = ensure_host_matvec_scratch(&context->impl, activation_bytes,
        static_cast<size_t>(weights->impl.rows) * sizeof(float));
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(context->impl.scratch_raw, device_activation, activation_bytes,
            cudaMemcpyDeviceToDevice, context->impl.stream);
    if (error == cudaSuccess) {
        launch_q8_repack_activation(context->impl.scratch_raw, context->impl.scratch_activation,
            weights->impl.blocks, context->impl.stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaEventRecord(context->impl.start, context->impl.stream);
    if (error == cudaSuccess) {
        launch_q8(const_cast<fortai_cuda_q8_weights_impl *>(&weights->impl),
            context->impl.scratch_activation,
            static_cast<float *>(device_output), context->impl.stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaEventRecord(context->impl.stop, context->impl.stream);
    if (error == cudaSuccess) error = cudaEventSynchronize(context->impl.stop);
    if (error == cudaSuccess) error = cudaEventElapsedTime(kernel_ms,
        context->impl.start, context->impl.stop);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "resident matvec", error);
}

extern "C" int fortai_cuda_q8_matvec_device_f32(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    if (!context || !weights || weights->impl.context != &context->impl ||
        !device_activation || !device_output || activation_elements !=
            static_cast<size_t>(weights->impl.blocks) * q8_block_width ||
        output_elements < static_cast<size_t>(weights->impl.rows))
        return FORTAI_CUDA_INVALID;
    const int blocks = weights->impl.blocks;
    cudaSetDevice(context->impl.device);
    cudaError_t error = ensure_device_matvec_scratch(&context->impl, activation_elements);
    if (error == cudaSuccess) {
        qwen_quantize_q8<<<(blocks + 31) / 32, 32, 0, context->impl.stream>>>(
            static_cast<const float *>(device_activation), context->impl.scratch_activation, blocks);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(const_cast<fortai_cuda_q8_weights_impl *>(&weights->impl),
            context->impl.scratch_activation, static_cast<float *>(device_output),
            context->impl.stream);
        error = cudaGetLastError();
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device F32 matvec", error);
}

extern "C" int fortai_cuda_q8_matvec_device_f32_pair(
    fortai_cuda_q8_context *context, const fortai_cuda_q8_weights *first_weights,
    const fortai_cuda_q8_weights *second_weights, const void *device_activation,
    size_t activation_elements, void *first_output, size_t first_output_elements,
    void *second_output, size_t second_output_elements) {
    if (!context || !first_weights || !second_weights ||
        first_weights->impl.context != &context->impl ||
        second_weights->impl.context != &context->impl ||
        !device_activation || !first_output || !second_output ||
        activation_elements != static_cast<size_t>(first_weights->impl.blocks) * q8_block_width ||
        first_weights->impl.blocks != second_weights->impl.blocks ||
        first_output_elements < static_cast<size_t>(first_weights->impl.rows) ||
        second_output_elements < static_cast<size_t>(second_weights->impl.rows)) {
        return FORTAI_CUDA_INVALID;
    }
    const int blocks = first_weights->impl.blocks;
    cudaSetDevice(context->impl.device);
    cudaError_t error = ensure_device_matvec_scratch(&context->impl, activation_elements);
    if (error == cudaSuccess) {
        qwen_quantize_q8<<<(blocks + 31) / 32, 32, 0, context->impl.stream>>>(
            static_cast<const float *>(device_activation), context->impl.scratch_activation, blocks);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(const_cast<fortai_cuda_q8_weights_impl *>(&first_weights->impl),
            context->impl.scratch_activation, static_cast<float *>(first_output),
            context->impl.stream);
        launch_q8(const_cast<fortai_cuda_q8_weights_impl *>(&second_weights->impl),
            context->impl.scratch_activation, static_cast<float *>(second_output),
            context->impl.stream);
        error = cudaGetLastError();
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device F32 matvec pair", error);
}

extern "C" int fortai_cuda_q8_reserve_matvec_scratch(
    fortai_cuda_q8_context *context, size_t activation_elements,
    size_t output_elements) {
    if (!context || activation_elements == 0 || output_elements == 0)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const size_t blocks = (activation_elements + q8_block_width - 1) / q8_block_width;
    cudaError_t error = ensure_host_matvec_scratch(&context->impl,
        blocks * q8_block_bytes, output_elements * sizeof(float));
    if (error == cudaSuccess)
        error = ensure_mmq_fixup_scratch(&context->impl,
            mmq_grid_blocks * mmq_tile_elements * sizeof(float),
            mmq_grid_blocks * sizeof(int32_t));
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "reserve device F32 matvec", error);
}

extern "C" int fortai_cuda_q8_matmul_device_f32(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *device_activation,
    size_t activation_elements, int batch, void *device_output,
    size_t output_elements) {
    if (!context || !weights || weights->impl.context != &context->impl ||
        !device_activation || !device_output || batch <= 0 ||
        batch > INT32_MAX || activation_elements !=
            static_cast<size_t>(weights->impl.blocks) * q8_block_width * batch ||
        output_elements < static_cast<size_t>(weights->impl.rows) * batch)
        return FORTAI_CUDA_INVALID;
    const int blocks = weights->impl.blocks;
    const int width = blocks * q8_block_width;
    const size_t packed_bytes = static_cast<size_t>(batch) * blocks * q8_activation_block_bytes;
    const bool use_mmq = weights->impl.rows >= 128 && blocks >= 8 &&
        batch >= 16 && blocks % 4 == 0;
    cudaSetDevice(context->impl.device);
    cudaError_t error = ensure_batched_activation_scratch(&context->impl, packed_bytes);
    if (error == cudaSuccess) {
        if (use_mmq) {
            qwen_quantize_q8_mmq_batch<<<dim3(static_cast<unsigned>(blocks / 4),
                static_cast<unsigned>(batch), 1), 128, 0, context->impl.stream>>>(
                static_cast<const float *>(device_activation), context->impl.scratch_batched_activation,
                width, blocks / 4, batch);
        } else {
            qwen_quantize_q8_batch<<<dim3(static_cast<unsigned>(blocks),
                static_cast<unsigned>(batch), 1), 32, 0, context->impl.stream>>>(
                static_cast<const float *>(device_activation), context->impl.scratch_batched_activation,
                width, blocks, batch);
        }
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        if (use_mmq) {
            constexpr int rows_per_tile = 128;
            constexpr int tokens_per_tile = 64;
            constexpr int blocks_per_tile = 8;
            constexpr size_t shared_bytes =
                rows_per_tile * (2 * q8_block_width + 1) * sizeof(int) +
                (rows_per_tile * blocks_per_tile + rows_per_tile / 4) * sizeof(float) +
                tokens_per_tile * (4 + q8_block_width) * sizeof(int);
            const int k_tiles = (blocks + blocks_per_tile - 1) / blocks_per_tile;
            const bool full_tiles = weights->impl.rows % rows_per_tile == 0 &&
                batch % tokens_per_tile == 0 && blocks % blocks_per_tile == 0;
            /* The bounds-aware MMA variant is also profitable for the final
             * short prompt chunk.  Llama.cpp keeps its MMA layout for these
             * tails; falling back to DP4A there left a disproportionate
             * amount of work on the critical path. */
            const bool mma_path = context->impl.mma_available &&
                weights->impl.rows >= rows_per_tile && batch >= 32 && blocks % blocks_per_tile == 0;
            /* The 128-token MMA specialization needs 254 registers/thread on
             * sm_120 and spills its accumulator frame to local memory.  Two
             * 64-token tiles keep the same tensor-core arithmetic without the
             * spill, and are faster for every production ubatch measured so
             * far.  Keep the wide kernel available for future tuning, but do
             * not select it on the serving path until it is genuinely faster. */
            const char *wide_env = std::getenv("FORTAI_CUDA_MMQ_WIDE");
            const bool wide_candidate = wide_env && wide_env[0] == '1';
            /* A 64-token tile keeps the launch count low but leaves 208
             * registers live per thread on sm_120.  The 32-token variant
             * keeps the same MMA instruction shape with four float4
             * accumulators, restoring occupancy without excessive tile
             * scheduling overhead. */
            const char *tokens_env = std::getenv("FORTAI_CUDA_MMQ_TOKENS");
            const int tile_tokens = (wide_candidate || (tokens_env && tokens_env[0] == '1')) ? 128 :
                (tokens_env && tokens_env[0] == '9' ? 96 : tokens_per_tile);
            const size_t mma_shared_bytes =
                static_cast<size_t>(tile_tokens) * sizeof(int) +
                static_cast<size_t>(tile_tokens) * (4 + q8_block_width) * sizeof(int) +
                static_cast<size_t>(rows_per_tile) * (2 * q8_block_width + 4 + 8) * sizeof(int);
            const int tile_count =
                ((weights->impl.rows + rows_per_tile - 1) / rows_per_tile) *
                ((batch + tile_tokens - 1) / tile_tokens);
            const int total_units = tile_count * k_tiles;
            /* With enough output tiles, give each tile one resident block and
             * reduce its K range locally.  This is the no-fixup case used by
             * llama.cpp when tiling is efficient: it avoids split-tile
             * scratch traffic and, importantly, gives every block a private
             * accumulator.  Keep stream-K for very short matrices where it
             * is needed to fill the SMs. */
            /* Stream-K's fixed resident grid keeps all SMs busy and matches
             * the proven llama.cpp work partition.  One-block-per-output-tile
             * scheduling looked attractive for avoiding fixup, but it creates
             * 112--192 tiny launches for this model and loses several times
             * more to occupancy and launch scheduling than it saves. */
            const char *stream_k_env = std::getenv("FORTAI_CUDA_MMQ_STREAM_K");
            const bool tile_parallel = !(stream_k_env && stream_k_env[0] == '1');
            const int grid_blocks = tile_parallel ? tile_count : min(total_units, mmq_grid_blocks);
            /* J=128 is llama.cpp's wide-token configuration on this device.
             * It halves the number of output tiles for full 128-token chunks,
             * but needs the opt-in 58 KiB dynamic shared-memory budget. */
            const bool wide_mma = false;
            const bool needs_fixup = mma_path && !tile_parallel && k_tiles > 1 &&
                tile_count % grid_blocks != 0;
            const bool needs_zero = !mma_path && k_tiles > 1;
            error = ensure_mmq_fixup_scratch(&context->impl,
                mmq_grid_blocks * static_cast<size_t>(tile_tokens) * rows_per_tile * sizeof(float),
                static_cast<size_t>(grid_blocks) * sizeof(int32_t));
            if (error == cudaSuccess && needs_zero)
                error = cudaMemsetAsync(device_output, 0,
                    static_cast<size_t>(weights->impl.rows) * batch * sizeof(float),
                    context->impl.stream);
            if (error == cudaSuccess) {
                /* Keep the steady-state tile on llama.cpp's bounds-free
                 * specialization.  A short final token tile is launched
                 * separately with the bounds-aware specialization; its
                 * activation stride remains the original batch stride, while
                 * its activation/output pointers are offset to the tail. */
                const bool split_token_tail = mma_path && tile_parallel &&
                    weights->impl.rows % rows_per_tile == 0 &&
                    blocks % blocks_per_tile == 0 && batch > tile_tokens &&
                    batch % tile_tokens != 0;
                if (split_token_tail) {
                    const int full_batch = (batch / tile_tokens) * tile_tokens;
                    const int tail_batch = batch - full_batch;
                    const int row_tiles_part = weights->impl.rows / rows_per_tile;
                    const size_t activation_stride = static_cast<size_t>(batch);
                    auto launch_mma_part = [&](int token_offset, int part_batch,
                        bool part_full) -> cudaError_t {
                        const int part_tiles = row_tiles_part *
                            ((part_batch + tile_tokens - 1) / tile_tokens);
                        const int part_grid = tile_parallel ? part_tiles :
                            min(part_tiles * k_tiles, mmq_grid_blocks);
                        const int8_t *part_activation =
                            context->impl.scratch_batched_activation +
                            static_cast<size_t>(token_offset) * (4 + q8_block_width);
                        float *part_output = static_cast<float *>(device_output) +
                            static_cast<size_t>(token_offset) * weights->impl.rows;
                        cudaError_t part_error = cudaSuccess;
                        if (tile_tokens == 128) {
                            if (part_full) {
                                part_error = cudaFuncSetAttribute(
                                    q8_gemm_batch_mmq_mma_layout<128, 128, 8, true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(mma_shared_bytes));
                                if (part_error == cudaSuccess)
                                    q8_gemm_batch_mmq_mma_layout<128, 128, 8, true>
                                        <<<part_grid, dim3(32, 8, 1), mma_shared_bytes,
                                            context->impl.stream>>>(
                                        weights->impl.device_data, part_activation,
                                        weights->impl.rows, blocks, part_batch,
                                        static_cast<int>(activation_stride), part_output,
                                        context->impl.scratch_mmq_fixup,
                                        context->impl.scratch_mmq_tiles);
                            } else {
                                part_error = cudaFuncSetAttribute(
                                    q8_gemm_batch_mmq_mma_layout<128, 128, 8, false>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(mma_shared_bytes));
                                if (part_error == cudaSuccess)
                                    q8_gemm_batch_mmq_mma_layout<128, 128, 8, false>
                                        <<<part_grid, dim3(32, 8, 1), mma_shared_bytes,
                                            context->impl.stream>>>(
                                        weights->impl.device_data, part_activation,
                                        weights->impl.rows, blocks, part_batch,
                                        static_cast<int>(activation_stride), part_output,
                                        context->impl.scratch_mmq_fixup,
                                        context->impl.scratch_mmq_tiles);
                            }
                        } else if (part_full) {
                            q8_gemm_batch_mmq_mma_layout<128, 64, 8, true>
                                <<<part_grid, dim3(32, 8, 1), mma_shared_bytes,
                                    context->impl.stream>>>(
                                weights->impl.device_data, part_activation,
                                weights->impl.rows, blocks, part_batch,
                                static_cast<int>(activation_stride), part_output,
                                context->impl.scratch_mmq_fixup,
                                context->impl.scratch_mmq_tiles);
                        } else {
                            q8_gemm_batch_mmq_mma_layout<128, 64, 8, false>
                                <<<part_grid, dim3(32, 8, 1), mma_shared_bytes,
                                    context->impl.stream>>>(
                                weights->impl.device_data, part_activation,
                                weights->impl.rows, blocks, part_batch,
                                static_cast<int>(activation_stride), part_output,
                                context->impl.scratch_mmq_fixup,
                                context->impl.scratch_mmq_tiles);
                        }
                        if (part_error != cudaSuccess) return part_error;
                        part_error = cudaGetLastError();
                        if (part_error != cudaSuccess) return part_error;
                        return cudaSuccess;
                    };
                    error = launch_mma_part(0, full_batch, true);
                    if (error == cudaSuccess)
                        error = launch_mma_part(full_batch, tail_batch, false);
                } else if (wide_mma) {
                    constexpr size_t wide_mma_shared_bytes = 81152;
                    const bool wide_full = weights->impl.rows % rows_per_tile == 0 &&
                        batch % 128 == 0 && blocks % blocks_per_tile == 0;
                    if (wide_full) {
                        error = cudaFuncSetAttribute(
                            q8_gemm_batch_mmq_mma_layout_wide_split<128, true>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            static_cast<int>(wide_mma_shared_bytes));
                        if (error == cudaSuccess)
                            q8_gemm_batch_mmq_mma_layout_wide_split<128, true><<<grid_blocks,
                                dim3(32, 8, 1), wide_mma_shared_bytes, context->impl.stream>>>(
                                weights->impl.device_data, context->impl.scratch_batched_activation,
                                weights->impl.rows, blocks, batch,
                                static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                                context->impl.scratch_mmq_tiles);
                    } else {
                        error = cudaFuncSetAttribute(
                            q8_gemm_batch_mmq_mma_layout_wide_split<128, false>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            static_cast<int>(wide_mma_shared_bytes));
                        if (error == cudaSuccess)
                            q8_gemm_batch_mmq_mma_layout_wide_split<128, false><<<grid_blocks,
                                dim3(32, 8, 1), wide_mma_shared_bytes, context->impl.stream>>>(
                                weights->impl.device_data, context->impl.scratch_batched_activation,
                                weights->impl.rows, blocks, batch,
                                static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                                context->impl.scratch_mmq_tiles);
                    }
                } else if (mma_path && full_tiles && tile_tokens == 64) {
                    if (tile_parallel) {
                        q8_gemm_batch_mmq_mma_layout<128, 64, 8, true><<<grid_blocks,
                            dim3(32, 8, 1), mma_shared_bytes, context->impl.stream>>>(
                            weights->impl.device_data, context->impl.scratch_batched_activation,
                            weights->impl.rows, blocks, batch, batch,
                            static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                            context->impl.scratch_mmq_tiles);
                    } else {
                        q8_gemm_batch_mmq_mma_layout<128, 64, 8, true><<<grid_blocks,
                            dim3(32, 8, 1), mma_shared_bytes, context->impl.stream>>>(
                            weights->impl.device_data, context->impl.scratch_batched_activation,
                            weights->impl.rows, blocks, batch, batch,
                            static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                            context->impl.scratch_mmq_tiles);
                    }
                } else if (mma_path) {
                    if (tile_tokens == 128) {
                        if (tile_parallel) {
                            error = cudaFuncSetAttribute(
                                q8_gemm_batch_mmq_mma_layout<128, 128, 8, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(mma_shared_bytes));
                            if (error == cudaSuccess)
                                q8_gemm_batch_mmq_mma_layout<128, 128, 8, false><<<grid_blocks,
                                    dim3(32, 8, 1), mma_shared_bytes, context->impl.stream>>>(
                                    weights->impl.device_data, context->impl.scratch_batched_activation,
                                    weights->impl.rows, blocks, batch, batch,
                                    static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                                    context->impl.scratch_mmq_tiles);
                        } else {
                            error = cudaFuncSetAttribute(
                                q8_gemm_batch_mmq_mma_layout<128, 128, 8, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(mma_shared_bytes));
                            if (error == cudaSuccess)
                                q8_gemm_batch_mmq_mma_layout<128, 128, 8, false><<<grid_blocks,
                                    dim3(32, 8, 1), mma_shared_bytes, context->impl.stream>>>(
                                    weights->impl.device_data, context->impl.scratch_batched_activation,
                                    weights->impl.rows, blocks, batch, batch,
                                    static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                                    context->impl.scratch_mmq_tiles);
                        }
                    } else if (tile_tokens == 96) {
                        if (tile_parallel) {
                            error = cudaFuncSetAttribute(
                                q8_gemm_batch_mmq_mma_layout<128, 96, 8, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(mma_shared_bytes));
                            if (error == cudaSuccess)
                                q8_gemm_batch_mmq_mma_layout<128, 96, 8, false><<<grid_blocks,
                                    dim3(32, 8, 1), mma_shared_bytes, context->impl.stream>>>(
                                    weights->impl.device_data, context->impl.scratch_batched_activation,
                                    weights->impl.rows, blocks, batch, batch,
                                    static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                                    context->impl.scratch_mmq_tiles);
                        } else {
                            error = cudaFuncSetAttribute(
                                q8_gemm_batch_mmq_mma_layout<128, 96, 8, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(mma_shared_bytes));
                            if (error == cudaSuccess)
                                q8_gemm_batch_mmq_mma_layout<128, 96, 8, false><<<grid_blocks,
                                    dim3(32, 8, 1), mma_shared_bytes, context->impl.stream>>>(
                                    weights->impl.device_data, context->impl.scratch_batched_activation,
                                    weights->impl.rows, blocks, batch, batch,
                                    static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                                    context->impl.scratch_mmq_tiles);
                        }
                    } else {
                        if (tile_parallel) {
                            q8_gemm_batch_mmq_mma_layout<128, 64, 8, false><<<grid_blocks,
                                dim3(32, 8, 1), mma_shared_bytes, context->impl.stream>>>(
                                weights->impl.device_data, context->impl.scratch_batched_activation,
                                weights->impl.rows, blocks, batch, batch,
                                static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                                context->impl.scratch_mmq_tiles);
                        } else {
                            q8_gemm_batch_mmq_mma_layout<128, 64, 8, false><<<grid_blocks,
                                dim3(32, 8, 1), mma_shared_bytes, context->impl.stream>>>(
                                weights->impl.device_data, context->impl.scratch_batched_activation,
                                weights->impl.rows, blocks, batch, batch,
                                static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                                context->impl.scratch_mmq_tiles);
                        }
                    }
                } else if (full_tiles) {
                    q8_gemm_batch_mmq_layout<128, 64, 8, true><<<grid_blocks,
                        dim3(32, 8, 1), shared_bytes, context->impl.stream>>>(
                        weights->impl.device_data, context->impl.scratch_batched_activation,
                        weights->impl.rows, blocks, batch,
                        static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                        context->impl.scratch_mmq_tiles);
                } else {
                    q8_gemm_batch_mmq_layout<128, 64, 8, false><<<grid_blocks,
                        dim3(32, 8, 1), shared_bytes, context->impl.stream>>>(
                        weights->impl.device_data, context->impl.scratch_batched_activation,
                        weights->impl.rows, blocks, batch,
                        static_cast<float *>(device_output), context->impl.scratch_mmq_fixup,
                        context->impl.scratch_mmq_tiles);
                }
                error = cudaGetLastError();
            }
            if (error == cudaSuccess && needs_fixup) {
                if (tile_tokens == 128) {
                    constexpr int fixup_x = 4;
                    q8_gemm_batch_mmq_fixup<128, 128><<<dim3(fixup_x, tile_count, 1), 256, 0,
                        context->impl.stream>>>(context->impl.scratch_mmq_fixup,
                        context->impl.scratch_mmq_tiles, weights->impl.rows, batch,
                        (weights->impl.rows + rows_per_tile - 1) / rows_per_tile,
                        tile_count, grid_blocks, static_cast<float *>(device_output));
                } else {
                    constexpr int fixup_x = 4;
                    q8_gemm_batch_mmq_fixup<128, 64><<<dim3(fixup_x, tile_count, 1), 256, 0,
                        context->impl.stream>>>(context->impl.scratch_mmq_fixup,
                        context->impl.scratch_mmq_tiles, weights->impl.rows, batch,
                        (weights->impl.rows + rows_per_tile - 1) / rows_per_tile,
                        tile_count, grid_blocks, static_cast<float *>(device_output));
                }
                error = cudaGetLastError();
            }
        } else {
            const int threads = weights->impl.blocks > 96 ? 128 :
                (weights->impl.blocks >= 32 ? 64 : 32);
            q8_gemm_batch<<<dim3(static_cast<unsigned>(weights->impl.rows),
                static_cast<unsigned>(batch), 1), threads, 0, context->impl.stream>>>(
                weights->impl.device_data, context->impl.scratch_batched_activation,
                weights->impl.rows, blocks, batch, static_cast<float *>(device_output));
        }
        if (!use_mmq) error = cudaGetLastError();
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device F32 batched matmul", error);
}

extern "C" int fortai_cuda_qwen35_embedding_device(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, int64_t token_id, void *device_output,
    size_t output_elements) {
    if (!context || !weights || weights->impl.context != &context->impl ||
        !device_output || token_id < 0 || token_id >= weights->impl.rows ||
        output_elements < static_cast<size_t>(weights->impl.width))
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    qwen_embedding_lookup_q8<<<(weights->impl.width + 255) / 256, 256, 0, context->impl.stream>>>(
        weights->impl.device_data, token_id, weights->impl.blocks,
        static_cast<float *>(device_output));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device Q8 embedding", error);
}

extern "C" int fortai_cuda_qwen35_embedding_device_batch(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const int32_t *host_tokens, int batch,
    void *device_output, size_t output_elements) {
    if (!context || !weights || weights->impl.context != &context->impl ||
        !host_tokens || batch <= 0 || batch > INT32_MAX || !device_output ||
        output_elements < static_cast<size_t>(weights->impl.width) * batch)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    cudaError_t error = ensure_token_scratch(&context->impl,
        static_cast<size_t>(batch) * sizeof(int32_t));
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(context->impl.scratch_tokens, host_tokens,
            static_cast<size_t>(batch) * sizeof(int32_t), cudaMemcpyHostToDevice,
            context->impl.stream);
    if (error == cudaSuccess) {
        qwen_embedding_lookup_q8_batch<<<dim3(
            static_cast<unsigned>((weights->impl.width + 255) / 256),
            static_cast<unsigned>(batch), 1), 256, 0, context->impl.stream>>>(
            weights->impl.device_data, context->impl.scratch_tokens, weights->impl.blocks,
            batch, static_cast<float *>(device_output));
        error = cudaGetLastError();
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device batched Q8 embedding", error);
}

extern "C" int fortai_cuda_qwen35_copy_device(fortai_cuda_q8_context *context,
    const void *device_input, void *device_output, size_t bytes) {
    if (!context || !device_input || !device_output || bytes == 0) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaMemcpyAsync(device_output, device_input, bytes,
        cudaMemcpyDeviceToDevice, context->impl.stream);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device copy", error);
}

extern "C" int fortai_cuda_qwen35_add_device(fortai_cuda_q8_context *context,
    const void *device_left, const void *device_right, void *device_output,
    size_t elements) {
    if (!context || !device_left || !device_right || !device_output || elements == 0 ||
        elements > static_cast<size_t>(INT32_MAX)) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    qwen_add_float<<<(static_cast<int>(elements) + 255) / 256, 256, 0,
        context->impl.stream>>>(static_cast<const float *>(device_left),
        static_cast<const float *>(device_right), static_cast<float *>(device_output),
        static_cast<int>(elements));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device add", error);
}

extern "C" int fortai_cuda_qwen35_copy_column_device(fortai_cuda_q8_context *context,
    const void *device_input, size_t stride, int column, void *device_output,
    size_t elements) {
    if (!context || !device_input || !device_output || stride == 0 ||
        column < 0 || column > INT32_MAX || elements == 0 || elements > stride ||
        stride > static_cast<size_t>(INT32_MAX)) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    qwen_copy_column<<<(static_cast<int>(elements) + 255) / 256, 256, 0,
        context->impl.stream>>>(static_cast<const float *>(device_input),
        static_cast<int>(stride), column, static_cast<float *>(device_output),
        static_cast<int>(elements));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device column copy", error);
}

extern "C" int fortai_cuda_qwen35_add_matrix_device(fortai_cuda_q8_context *context,
    const void *device_left, const void *device_right, void *device_output,
    size_t elements) {
    if (!context || !device_left || !device_right || !device_output || elements == 0 ||
        elements > static_cast<size_t>(INT32_MAX)) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    qwen_add_matrix<<<(static_cast<int>(elements) + 255) / 256, 256, 0,
        context->impl.stream>>>(static_cast<const float *>(device_left),
        static_cast<const float *>(device_right), static_cast<float *>(device_output),
        static_cast<int>(elements));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device matrix add", error);
}

extern "C" int fortai_cuda_qwen35_silu_product_device(fortai_cuda_q8_context *context,
    void *device_gate, const void *device_up, size_t elements) {
    if (!context || !device_gate || !device_up || elements == 0 ||
        elements > static_cast<size_t>(INT32_MAX)) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    qwen_silu_product<<<(static_cast<int>(elements) + 255) / 256, 256,
        0, context->impl.stream>>>(static_cast<float *>(device_gate),
        static_cast<const float *>(device_up), static_cast<int>(elements));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device SiLU product", error);
}

extern "C" int fortai_cuda_qwen35_silu_product_matrix_device(
    fortai_cuda_q8_context *context, void *device_gate, const void *device_up,
    size_t elements) {
    if (!context || !device_gate || !device_up || elements == 0 ||
        elements > static_cast<size_t>(INT32_MAX)) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    qwen_silu_product_matrix<<<(static_cast<int>(elements) + 255) / 256, 256,
        0, context->impl.stream>>>(static_cast<float *>(device_gate),
        static_cast<const float *>(device_up), static_cast<int>(elements));
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device matrix SiLU product", error);
}

extern "C" int fortai_cuda_qwen35_rms_norm_device(fortai_cuda_q8_context *context,
    const void *device_input, const void *device_weights, void *device_output,
    size_t elements, float epsilon) {
    if (!context || !device_input || !device_weights || !device_output || elements == 0 ||
        elements > static_cast<size_t>(INT32_MAX) || epsilon <= 0.0f)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const char *fast_env = std::getenv("FORTAI_CUDA_RMS_NORM_FP32");
    const bool fast = fast_env != nullptr && (fast_env[0] == '1' || fast_env[0] == 'y' ||
        fast_env[0] == 'Y' || fast_env[0] == 't' || fast_env[0] == 'T');
    if (fast) {
        qwen_rms_norm_float_fast<<<1, 1024, 0, context->impl.stream>>>(
            static_cast<const float *>(device_input), static_cast<const float *>(device_weights),
            static_cast<float *>(device_output), static_cast<int>(elements), epsilon);
    } else {
        qwen_rms_norm_float<<<1, 1024, 0, context->impl.stream>>>(
            static_cast<const float *>(device_input), static_cast<const float *>(device_weights),
            static_cast<float *>(device_output), static_cast<int>(elements), epsilon);
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device RMS norm", error);
}

extern "C" int fortai_cuda_qwen35_rms_norm_matrix_device(
    fortai_cuda_q8_context *context, const void *device_input,
    const void *device_weights, void *device_output, size_t hidden, int batch,
    float epsilon) {
    if (!context || !device_input || !device_weights || !device_output ||
        hidden == 0 || hidden > static_cast<size_t>(INT32_MAX) || batch <= 0 ||
        batch > INT32_MAX || epsilon <= 0.0f) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    qwen_rms_norm_matrix<<<static_cast<unsigned>(batch), 1024, 0,
        context->impl.stream>>>(static_cast<const float *>(device_input),
        static_cast<const float *>(device_weights), static_cast<float *>(device_output),
        static_cast<int>(hidden), batch, epsilon);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device batched RMS norm", error);
}

extern "C" int fortai_cuda_qwen35_argmax_device(fortai_cuda_q8_context *context,
    const void *device_logits, size_t elements, int *host_index) {
    if (!context || !device_logits || !host_index || elements == 0 ||
        elements > static_cast<size_t>(INT32_MAX)) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const int threads = 256;
    const int blocks = std::min<int>(static_cast<int>((elements + threads - 1) / threads), 1024);
    const size_t bytes = static_cast<size_t>(blocks) * sizeof(qwen_argmax_pair);
    cudaError_t error = ensure_aux_scratch(&context->impl, bytes);
    if (error == cudaSuccess) {
        qwen_argmax_partials<<<blocks, threads, threads * sizeof(qwen_argmax_pair),
            context->impl.stream>>>(static_cast<const float *>(device_logits),
            static_cast<int>(elements), reinterpret_cast<qwen_argmax_pair *>(context->impl.scratch_aux));
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = ensure_download_host(&context->impl, bytes);
    auto *partials = static_cast<qwen_argmax_pair *>(context->impl.download_host);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(partials, context->impl.scratch_aux, bytes,
            cudaMemcpyDeviceToHost, context->impl.stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->impl.stream);
    if (error != cudaSuccess)
        return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device argmax", error);
    qwen_argmax_pair best = {-FLT_MAX, INT_MAX};
    for (int block = 0; block < blocks; ++block) {
        const qwen_argmax_pair candidate = partials[block];
        if (candidate.value > best.value ||
            (candidate.value == best.value && candidate.index < best.index)) best = candidate;
    }
    *host_index = best.index;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_qwen35_argmax_rows_device(fortai_cuda_q8_context *context,
    const void *device_logits, size_t row_elements, int rows, int *host_indices) {
    if (!context || !device_logits || !host_indices || row_elements == 0 ||
        row_elements > static_cast<size_t>(INT32_MAX) || rows <= 0 || rows > 32)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const int threads = 256;
    const int blocks = std::min<int>(
        static_cast<int>((row_elements + threads - 1) / threads), 1024);
    const size_t pairs = static_cast<size_t>(blocks) * rows;
    const size_t bytes = pairs * sizeof(qwen_argmax_pair);
    cudaError_t error = ensure_aux_scratch(&context->impl, bytes);
    if (error == cudaSuccess) {
        qwen_argmax_rows_partials<<<dim3(static_cast<unsigned>(blocks),
            static_cast<unsigned>(rows), 1), threads,
            threads * sizeof(qwen_argmax_pair), context->impl.stream>>>(
            static_cast<const float *>(device_logits), static_cast<int>(row_elements),
            reinterpret_cast<qwen_argmax_pair *>(context->impl.scratch_aux));
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = ensure_download_host(&context->impl, bytes);
    auto *partials = static_cast<qwen_argmax_pair *>(context->impl.download_host);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(partials, context->impl.scratch_aux, bytes,
            cudaMemcpyDeviceToHost, context->impl.stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->impl.stream);
    if (error != cudaSuccess)
        return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device row argmax", error);
    for (int row = 0; row < rows; ++row) {
        qwen_argmax_pair best = {-FLT_MAX, INT_MAX};
        for (int block = 0; block < blocks; ++block) {
            const qwen_argmax_pair candidate =
                partials[static_cast<size_t>(row) * blocks + block];
            if (candidate.value > best.value ||
                (candidate.value == best.value && candidate.index < best.index)) best = candidate;
        }
        host_indices[row] = best.index;
    }
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_qwen35_topk_rows_device(fortai_cuda_q8_context *context,
    const void *device_logits, size_t row_elements, int rows, int top_k,
    int *host_indices, float *host_values) {
    if (!context || !device_logits || !host_indices || !host_values ||
        row_elements == 0 || row_elements > static_cast<size_t>(INT32_MAX) ||
        rows <= 0 || rows > 32 || top_k <= 0 || top_k > qwen_topk_max ||
        static_cast<size_t>(top_k) > row_elements) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);

    if (rows <= 3) {
        const int status = qwen_topk_radix_rows(&context->impl,
            static_cast<const float *>(device_logits), static_cast<int>(row_elements),
            rows, top_k, host_indices, host_values);
        if (status >= 0) return status;
    }

    const size_t selected_count = static_cast<size_t>(rows) * top_k;
    auto requirements = cuda::execution::require(
        cuda::execution::determinism::not_guaranteed,
        cuda::execution::output_ordering::unsorted);
    auto stream_env = cuda::stream_ref{context->impl.stream};
    auto environment = cuda::std::execution::env{stream_env, requirements};
    auto indexes_in = cuda::make_counting_iterator(0);
    cudaError_t error = cudaSuccess;
    if (context->impl.topk_temp_elements != static_cast<int>(row_elements) ||
        context->impl.topk_temp_k != top_k) {
        size_t required = 0;
        error = cub::DeviceTopK::MaxPairs(nullptr, required,
            static_cast<const float *>(device_logits), static_cast<float *>(nullptr),
            indexes_in, static_cast<int *>(nullptr), static_cast<int>(row_elements),
            top_k, environment);
        if (error == cudaSuccess) {
            context->impl.topk_temp_bytes = required;
            context->impl.topk_temp_elements = static_cast<int>(row_elements);
            context->impl.topk_temp_k = top_k;
        }
    }
    const size_t temp_bytes = (context->impl.topk_temp_bytes + 255u) & ~size_t(255u);
    const size_t temp_copies = rows > 1 ? static_cast<size_t>(rows) : 1u;
    const size_t values_bytes = selected_count * sizeof(float);
    const size_t indices_bytes = selected_count * sizeof(int);
    if (error == cudaSuccess)
        error = ensure_aux_scratch(&context->impl,
            temp_copies * temp_bytes + values_bytes + indices_bytes);
    auto *device_values = reinterpret_cast<float *>(
        reinterpret_cast<unsigned char *>(context->impl.scratch_aux) + temp_copies * temp_bytes);
    auto *device_indices = reinterpret_cast<int *>(
        reinterpret_cast<unsigned char *>(device_values) + values_bytes);
    if (error == cudaSuccess)
        error = ensure_download_host(&context->impl, values_bytes + indices_bytes);
    auto *selected_values = static_cast<float *>(context->impl.download_host);
    auto *selected_indices = reinterpret_cast<int *>(
        reinterpret_cast<unsigned char *>(context->impl.download_host) + values_bytes);
    if (error == cudaSuccess && rows == 1) {
        size_t available = context->impl.topk_temp_bytes;
        error = cub::DeviceTopK::MaxPairs(context->impl.scratch_aux, available,
            static_cast<const float *>(device_logits), device_values, indexes_in,
            device_indices, static_cast<int>(row_elements), top_k, environment);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(selected_values, device_values, values_bytes,
                cudaMemcpyDeviceToHost, context->impl.stream);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(selected_indices, device_indices, indices_bytes,
                cudaMemcpyDeviceToHost, context->impl.stream);
        if (error == cudaSuccess) error = cudaStreamSynchronize(context->impl.stream);
    } else if (error == cudaSuccess) {
        if (!context->impl.topk_ready) {
            error = cudaEventCreateWithFlags(&context->impl.topk_ready, cudaEventDisableTiming);
            for (int row = 0; error == cudaSuccess && row < 32; ++row) {
                error = cudaStreamCreateWithFlags(&context->impl.topk_streams[row],
                    cudaStreamNonBlocking);
                if (error == cudaSuccess)
                    error = cudaEventCreateWithFlags(&context->impl.topk_done[row],
                        cudaEventDisableTiming);
            }
            if (error != cudaSuccess) {
                for (int row = 0; row < 32; ++row) {
                    if (context->impl.topk_done[row]) {
                        cudaEventDestroy(context->impl.topk_done[row]);
                        context->impl.topk_done[row] = nullptr;
                    }
                    if (context->impl.topk_streams[row]) {
                        cudaStreamDestroy(context->impl.topk_streams[row]);
                        context->impl.topk_streams[row] = nullptr;
                    }
                }
                if (context->impl.topk_ready) {
                    cudaEventDestroy(context->impl.topk_ready);
                    context->impl.topk_ready = nullptr;
                }
            }
        }
        if (error == cudaSuccess)
            error = cudaEventRecord(context->impl.topk_ready, context->impl.stream);
        for (int row = 0; error == cudaSuccess && row < rows; ++row) {
            cudaStream_t row_stream = context->impl.topk_streams[row];
            error = cudaStreamWaitEvent(row_stream, context->impl.topk_ready, 0);
            auto row_stream_env = cuda::stream_ref{row_stream};
            auto row_environment = cuda::std::execution::env{row_stream_env, requirements};
            size_t available = context->impl.topk_temp_bytes;
            if (error == cudaSuccess)
                error = cub::DeviceTopK::MaxPairs(
                    reinterpret_cast<unsigned char *>(context->impl.scratch_aux) +
                        static_cast<size_t>(row) * temp_bytes,
                    available,
                    static_cast<const float *>(device_logits) +
                        static_cast<size_t>(row) * row_elements,
                    device_values + static_cast<size_t>(row) * top_k, indexes_in,
                    device_indices + static_cast<size_t>(row) * top_k,
                    static_cast<int>(row_elements), top_k, row_environment);
            const size_t row_bytes_values = static_cast<size_t>(top_k) * sizeof(float);
            const size_t row_bytes_indices = static_cast<size_t>(top_k) * sizeof(int);
            if (error == cudaSuccess)
                error = cudaMemcpyAsync(selected_values + static_cast<size_t>(row) * top_k,
                    device_values + static_cast<size_t>(row) * top_k, row_bytes_values,
                    cudaMemcpyDeviceToHost, row_stream);
            if (error == cudaSuccess)
                error = cudaMemcpyAsync(selected_indices + static_cast<size_t>(row) * top_k,
                    device_indices + static_cast<size_t>(row) * top_k, row_bytes_indices,
                    cudaMemcpyDeviceToHost, row_stream);
            if (error == cudaSuccess)
                error = cudaEventRecord(context->impl.topk_done[row], row_stream);
        }
        for (int row = 0; error == cudaSuccess && row < rows; ++row)
            error = cudaEventSynchronize(context->impl.topk_done[row]);
    }
    if (error != cudaSuccess)
        return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device row top-k", error);
    std::vector<qwen_argmax_pair> ordered(static_cast<size_t>(top_k));
    for (int row = 0; row < rows; ++row) {
        const size_t base = static_cast<size_t>(row) * top_k;
        for (int i = 0; i < top_k; ++i)
            ordered[static_cast<size_t>(i)] = {
                selected_values[base + i], selected_indices[base + i]};
        std::sort(ordered.begin(), ordered.end(), [](const qwen_argmax_pair &left,
                const qwen_argmax_pair &right) {
            return left.value > right.value ||
                (left.value == right.value && left.index < right.index);
        });
        for (int i = 0; i < top_k; ++i) {
            host_indices[base + i] = ordered[static_cast<size_t>(i)].index;
            host_values[base + i] = ordered[static_cast<size_t>(i)].value;
        }
    }
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_qwen35_concat_device(fortai_cuda_q8_context *context,
    const void *device_first, const void *device_second, size_t elements,
    void *device_output) {
    if (!context || !device_first || !device_second || !device_output ||
        elements == 0) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    qwen_concat_float<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
        context->impl.stream>>>(static_cast<const float *>(device_first),
        static_cast<const float *>(device_second), static_cast<float *>(device_output), elements);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device concat", error);
}

extern "C" int fortai_cuda_qwen35_concat_matrix_device(fortai_cuda_q8_context *context,
    const void *device_first, const void *device_second, size_t hidden, int batch,
    void *device_output) {
    if (!context || !device_first || !device_second || !device_output || hidden == 0 ||
        hidden > static_cast<size_t>(INT32_MAX) || batch <= 0 || batch > INT32_MAX)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const size_t elements = hidden * static_cast<size_t>(batch);
    qwen_concat_float_matrix<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
        context->impl.stream>>>(static_cast<const float *>(device_first),
        static_cast<const float *>(device_second), static_cast<float *>(device_output),
        static_cast<int>(hidden), batch);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device batched concat", error);
}

extern "C" int fortai_cuda_qwen35_shift_target_hidden_device(
    fortai_cuda_q8_context *context, const void *device_input, void *device_pending,
    size_t hidden, int batch, void *device_output) {
    if (!context || !device_input || !device_pending || !device_output || hidden == 0 ||
        hidden > static_cast<size_t>(INT32_MAX) || batch <= 0 || batch > INT32_MAX)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const size_t elements = hidden * static_cast<size_t>(batch);
    qwen_shift_target_hidden<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
        context->impl.stream>>>(static_cast<const float *>(device_input),
        static_cast<float *>(device_pending), static_cast<float *>(device_output),
        static_cast<int>(hidden), batch);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device target-hidden shift", error);
}

extern "C" int fortai_cuda_q8_matvec_host(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t output_bytes,
    float *elapsed_ms) {
    if (!context || !weights || weights->impl.context != &context->impl ||
        validate_weights(&weights->impl, host_activation, activation_bytes,
            host_output, output_bytes) != FORTAI_CUDA_OK || !elapsed_ms)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const size_t output_size = static_cast<size_t>(weights->impl.rows) * sizeof(float);
    const size_t activation_size = static_cast<size_t>(weights->impl.blocks) * q8_block_bytes;
    cudaError_t error = ensure_host_matvec_scratch(&context->impl, activation_size, output_size);
    if (error == cudaSuccess) error = cudaEventRecord(context->impl.start, context->impl.stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(context->impl.scratch_raw, host_activation,
            activation_size,
            cudaMemcpyHostToDevice, context->impl.stream);
    if (error == cudaSuccess) {
        launch_q8_repack_activation(context->impl.scratch_raw, context->impl.scratch_activation,
            weights->impl.blocks, context->impl.stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(const_cast<fortai_cuda_q8_weights_impl *>(&weights->impl),
            context->impl.scratch_activation, context->impl.scratch_output,
            context->impl.stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(host_output, context->impl.scratch_output, output_size,
            cudaMemcpyDeviceToHost, context->impl.stream);
    if (error == cudaSuccess) error = cudaEventRecord(context->impl.stop, context->impl.stream);
    if (error == cudaSuccess) error = cudaEventSynchronize(context->impl.stop);
    if (error == cudaSuccess) error = cudaEventElapsedTime(elapsed_ms,
        context->impl.start, context->impl.stop);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "host matvec", error);
}

extern "C" int fortai_cuda_q8_matvec_host_pair(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *first_weights,
    const fortai_cuda_q8_weights *second_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *elapsed_ms) {
    const host_matvec_request requests[] = {
        {first_weights ? const_cast<fortai_cuda_q8_weights_impl *>(&first_weights->impl) : nullptr,
            first_output, first_output_bytes},
        {second_weights ? const_cast<fortai_cuda_q8_weights_impl *>(&second_weights->impl) : nullptr,
            second_output, second_output_bytes}
    };
    return matvec_host_many(context ? &context->impl : nullptr, requests, 2,
        host_activation, activation_bytes, elapsed_ms, nullptr);
}

extern "C" int fortai_cuda_q8_matvec_host_triplet(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *first_weights,
    const fortai_cuda_q8_weights *second_weights,
    const fortai_cuda_q8_weights *third_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *third_output,
    size_t third_output_bytes, float *elapsed_ms) {
    const host_matvec_request requests[] = {
        {first_weights ? const_cast<fortai_cuda_q8_weights_impl *>(&first_weights->impl) : nullptr,
            first_output, first_output_bytes},
        {second_weights ? const_cast<fortai_cuda_q8_weights_impl *>(&second_weights->impl) : nullptr,
            second_output, second_output_bytes},
        {third_weights ? const_cast<fortai_cuda_q8_weights_impl *>(&third_weights->impl) : nullptr,
            third_output, third_output_bytes}
    };
    return matvec_host_many(context ? &context->impl : nullptr, requests, 3,
        host_activation, activation_bytes, elapsed_ms, nullptr);
}

extern "C" int fortai_cuda_q8_matvec_host_triplet_contiguous(
    fortai_cuda_q8_context *context, const fortai_cuda_q8_weights *first_weights,
    const fortai_cuda_q8_weights *second_weights,
    const fortai_cuda_q8_weights *third_weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t host_output_bytes,
    float *elapsed_ms) {
    if (!first_weights || !second_weights || !third_weights || !host_output)
        return FORTAI_CUDA_INVALID;
    size_t first_bytes = static_cast<size_t>(first_weights->impl.rows) * sizeof(float);
    size_t second_bytes = static_cast<size_t>(second_weights->impl.rows) * sizeof(float);
    size_t third_bytes = static_cast<size_t>(third_weights->impl.rows) * sizeof(float);
    if (host_output_bytes < first_bytes + second_bytes + third_bytes)
        return FORTAI_CUDA_INVALID;
    const host_matvec_request requests[] = {
        {const_cast<fortai_cuda_q8_weights_impl *>(&first_weights->impl), host_output, first_bytes},
        {const_cast<fortai_cuda_q8_weights_impl *>(&second_weights->impl),
            host_output + first_weights->impl.rows, second_bytes},
        {const_cast<fortai_cuda_q8_weights_impl *>(&third_weights->impl),
            host_output + first_weights->impl.rows + second_weights->impl.rows, third_bytes}
    };
    return matvec_host_many(context ? &context->impl : nullptr, requests, 3,
        host_activation, activation_bytes, elapsed_ms, host_output);
}

extern "C" int fortai_cuda_q8_ffn_host(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *gate_weights,
    const fortai_cuda_q8_weights *up_weights,
    const fortai_cuda_q8_weights *down_weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t output_bytes,
    float *elapsed_ms) {
    if (!context || !gate_weights || !up_weights || !down_weights || !host_activation ||
        !host_output || !elapsed_ms)
        return FORTAI_CUDA_INVALID;
    auto *gate = const_cast<fortai_cuda_q8_weights_impl *>(&gate_weights->impl);
    auto *up = const_cast<fortai_cuda_q8_weights_impl *>(&up_weights->impl);
    auto *down = const_cast<fortai_cuda_q8_weights_impl *>(&down_weights->impl);
    if (gate->context != &context->impl || up->context != &context->impl ||
        down->context != &context->impl || gate->blocks != up->blocks ||
        gate->rows != up->rows || gate->rows != down->blocks * q8_block_width ||
        !gate->device_data || !up->device_data || !down->device_data)
        return FORTAI_CUDA_INVALID;
    const size_t expected_activation = static_cast<size_t>(gate->blocks) * q8_block_bytes;
    const size_t down_activation = static_cast<size_t>(down->blocks) * q8_block_bytes;
    const size_t gate_bytes = static_cast<size_t>(gate->rows) * sizeof(float);
    const size_t expected_output = static_cast<size_t>(down->rows) * sizeof(float);
    if (activation_bytes < expected_activation || output_bytes < expected_output)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    cudaError_t error = ensure_host_matvec_scratch(&context->impl,
        std::max(expected_activation, down_activation),
        std::max(gate_bytes, expected_output));
    if (error == cudaSuccess && gate->blocks >= 128)
        error = ensure_aux_scratch(&context->impl, gate_bytes);
    if (error == cudaSuccess) error = cudaEventRecord(context->impl.start, context->impl.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(context->impl.scratch_raw, host_activation,
        expected_activation, cudaMemcpyHostToDevice, context->impl.stream);
    if (error == cudaSuccess) {
        launch_q8_repack_activation(context->impl.scratch_raw, context->impl.scratch_activation,
            gate->blocks, context->impl.stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8_silu_product(gate, up, context->impl.scratch_activation,
            context->impl.scratch_output, context->impl.stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_quantize_q8<<<(down->blocks + 31) / 32, 32, 0, context->impl.stream>>>(
            context->impl.scratch_output, reinterpret_cast<int8_t *>(context->impl.scratch_activation),
            down->blocks);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(down, context->impl.scratch_activation, context->impl.scratch_output,
            context->impl.stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaMemcpyAsync(host_output, context->impl.scratch_output,
        expected_output, cudaMemcpyDeviceToHost, context->impl.stream);
    if (error == cudaSuccess) error = cudaEventRecord(context->impl.stop, context->impl.stream);
    if (error == cudaSuccess) error = cudaEventSynchronize(context->impl.stop);
    if (error == cudaSuccess) error = cudaEventElapsedTime(elapsed_ms,
        context->impl.start, context->impl.stop);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "Qwen FFN host", error);
}

extern "C" int fortai_cuda_q8_ffn_device(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *gate_weights,
    const fortai_cuda_q8_weights *up_weights,
    const fortai_cuda_q8_weights *down_weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    if (!context || !gate_weights || !up_weights || !down_weights || !device_activation ||
        !device_output) return FORTAI_CUDA_INVALID;
    auto *gate = const_cast<fortai_cuda_q8_weights_impl *>(&gate_weights->impl);
    auto *up = const_cast<fortai_cuda_q8_weights_impl *>(&up_weights->impl);
    auto *down = const_cast<fortai_cuda_q8_weights_impl *>(&down_weights->impl);
    if (gate->context != &context->impl || up->context != &context->impl ||
        down->context != &context->impl || gate->blocks != up->blocks ||
        gate->rows != up->rows || gate->rows != down->blocks * q8_block_width ||
        !gate->device_data || !up->device_data || !down->device_data ||
        activation_elements != static_cast<size_t>(gate->blocks) * q8_block_width ||
        output_elements < static_cast<size_t>(down->rows)) return FORTAI_CUDA_INVALID;
    const size_t activation_bytes = static_cast<size_t>(gate->blocks) * q8_block_bytes;
    const size_t down_activation_bytes = static_cast<size_t>(down->blocks) * q8_block_bytes;
    const size_t gate_bytes = static_cast<size_t>(gate->rows) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(down->rows) * sizeof(float);
    cudaSetDevice(context->impl.device);
    cudaError_t error = ensure_host_matvec_scratch(&context->impl,
        std::max(activation_bytes, down_activation_bytes),
        std::max(gate_bytes, output_bytes));
    if (error == cudaSuccess && gate->blocks >= 128)
        error = ensure_aux_scratch(&context->impl, gate_bytes);
    if (error == cudaSuccess) {
        qwen_quantize_q8<<<(gate->blocks + 31) / 32, 32, 0, context->impl.stream>>>(
            static_cast<const float *>(device_activation), context->impl.scratch_activation,
            gate->blocks);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8_silu_product(gate, up, context->impl.scratch_activation,
            context->impl.scratch_output, context->impl.stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_quantize_q8<<<(down->blocks + 31) / 32, 32, 0, context->impl.stream>>>(
            context->impl.scratch_output, context->impl.scratch_activation, down->blocks);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(down, context->impl.scratch_activation,
            static_cast<float *>(device_output), context->impl.stream);
        error = cudaGetLastError();
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "Qwen FFN device", error);
}

extern "C" int fortai_cuda_qwen35_recurrent_create(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *qkv_weights,
    const fortai_cuda_q8_weights *gate_weights,
    const fortai_cuda_q8_weights *alpha_weights,
    const fortai_cuda_q8_weights *beta_weights,
    const fortai_cuda_q8_weights *output_weights,
    const void *conv_weights, size_t conv_weight_bytes, int conv_size, int conv_kernel,
    const void *ssm_a, size_t ssm_a_bytes, const void *ssm_dt, size_t ssm_dt_bytes,
    const void *ssm_norm, size_t ssm_norm_bytes, int state_size, int key_heads,
    int value_heads, int head_size, int inner_size, float norm_epsilon,
    fortai_cuda_qwen35_recurrent **layer) {
    if (!context || !layer || !qkv_weights || !gate_weights || !alpha_weights ||
        !beta_weights || !output_weights || !conv_weights || !ssm_a || !ssm_dt ||
        !ssm_norm || conv_size <= 0 || conv_kernel < 2 || state_size <= 0 ||
        key_heads <= 0 || value_heads <= 0 || head_size <= 0 || inner_size <= 0 ||
        value_heads % key_heads != 0 || value_heads * head_size != inner_size)
        return FORTAI_CUDA_INVALID;
    *layer = nullptr;
    auto *created = new (std::nothrow) fortai_cuda_qwen35_recurrent;
    if (!created) return FORTAI_CUDA_RUNTIME_ERROR;
    created->impl.context = &context->impl;
    created->impl.qkv_weights = const_cast<fortai_cuda_q8_weights_impl *>(&qkv_weights->impl);
    created->impl.gate_weights = const_cast<fortai_cuda_q8_weights_impl *>(&gate_weights->impl);
    created->impl.alpha_weights = const_cast<fortai_cuda_q8_weights_impl *>(&alpha_weights->impl);
    created->impl.beta_weights = const_cast<fortai_cuda_q8_weights_impl *>(&beta_weights->impl);
    created->impl.output_weights = const_cast<fortai_cuda_q8_weights_impl *>(&output_weights->impl);
    if (created->impl.qkv_weights->context != &context->impl ||
        created->impl.gate_weights->context != &context->impl ||
        created->impl.alpha_weights->context != &context->impl ||
        created->impl.beta_weights->context != &context->impl ||
        created->impl.output_weights->context != &context->impl)
        goto invalid;
    created->impl.conv_size = conv_size;
    created->impl.conv_kernel = conv_kernel;
    created->impl.state_size = state_size;
    created->impl.key_heads = key_heads;
    created->impl.value_heads = value_heads;
    created->impl.head_size = head_size;
    created->impl.inner_size = inner_size;
    created->impl.norm_epsilon = norm_epsilon;
    if (conv_weight_bytes < static_cast<size_t>(conv_size) * conv_kernel * sizeof(float) ||
        ssm_a_bytes < static_cast<size_t>(value_heads) * sizeof(float) ||
        ssm_dt_bytes < static_cast<size_t>(value_heads) * sizeof(float) ||
        ssm_norm_bytes < static_cast<size_t>(head_size) * sizeof(float))
        goto invalid;
    cudaSetDevice(context->impl.device);
    {
        cudaError_t error = allocate_and_copy_float(&created->impl, &created->impl.conv_weights,
            conv_weights, static_cast<size_t>(conv_size) * conv_kernel * sizeof(float));
        if (error == cudaSuccess) error = allocate_and_copy_float(&created->impl, &created->impl.ssm_a,
            ssm_a, static_cast<size_t>(value_heads) * sizeof(float));
        if (error == cudaSuccess) error = allocate_and_copy_float(&created->impl, &created->impl.ssm_dt,
            ssm_dt, static_cast<size_t>(value_heads) * sizeof(float));
        if (error == cudaSuccess) error = allocate_and_copy_float(&created->impl, &created->impl.ssm_norm,
            ssm_norm, static_cast<size_t>(head_size) * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.conv_state),
            static_cast<size_t>(conv_kernel - 1) * conv_size * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.gdn_state),
            static_cast<size_t>(value_heads) * head_size * head_size * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.conv_state_first),
            static_cast<size_t>(conv_kernel - 1) * conv_size * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.gdn_state_first),
            static_cast<size_t>(value_heads) * head_size * head_size * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.conv_state_second),
            static_cast<size_t>(conv_kernel - 1) * conv_size * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.gdn_state_second),
            static_cast<size_t>(value_heads) * head_size * head_size * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.qkv_output),
            static_cast<size_t>(conv_size) * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.gate_output),
            static_cast<size_t>(inner_size) * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.alpha_output),
            static_cast<size_t>(value_heads) * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.beta_output),
            static_cast<size_t>(value_heads) * sizeof(float));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.gdn_output),
            static_cast<size_t>(inner_size) * sizeof(float));
        if (error == cudaSuccess) error = cudaMemsetAsync(created->impl.conv_state, 0,
            static_cast<size_t>(conv_kernel - 1) * conv_size * sizeof(float), context->impl.stream);
        if (error == cudaSuccess) error = cudaMemsetAsync(created->impl.gdn_state, 0,
            static_cast<size_t>(value_heads) * head_size * head_size * sizeof(float), context->impl.stream);
        /* State initialization is queued on the context stream.  The model
         * setup path performs one fence after all layers are created; a
         * per-layer synchronize here needlessly serializes large GGUF loads. */
        if (error != cudaSuccess) {
            free_recurrent_device_buffers(&created->impl);
            delete created;
            return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "Qwen recurrent create", error);
        }
    }
    *layer = created;
    return FORTAI_CUDA_OK;
invalid:
    delete created;
    return FORTAI_CUDA_INVALID;
}

extern "C" int fortai_cuda_qwen35_recurrent_create_state(
    fortai_cuda_q8_context *context, const void *conv_weights, size_t conv_weight_bytes,
    int conv_size, int conv_kernel, const void *ssm_a, size_t ssm_a_bytes,
    const void *ssm_dt, size_t ssm_dt_bytes, const void *ssm_norm,
    size_t ssm_norm_bytes, int state_size, int key_heads, int value_heads,
    int head_size, int inner_size, float norm_epsilon,
    fortai_cuda_qwen35_recurrent **layer) {
    if (!context || !layer || !conv_weights || !ssm_a || !ssm_dt || !ssm_norm ||
        conv_size <= 0 || conv_kernel < 2 || state_size <= 0 || key_heads <= 0 ||
        value_heads <= 0 || head_size <= 0 || inner_size <= 0 ||
        value_heads % key_heads != 0 || value_heads * head_size != inner_size ||
        norm_epsilon <= 0.0f)
        return FORTAI_CUDA_INVALID;
    if (conv_weight_bytes < static_cast<size_t>(conv_size) * conv_kernel * sizeof(float) ||
        ssm_a_bytes < static_cast<size_t>(value_heads) * sizeof(float) ||
        ssm_dt_bytes < static_cast<size_t>(value_heads) * sizeof(float) ||
        ssm_norm_bytes < static_cast<size_t>(head_size) * sizeof(float))
        return FORTAI_CUDA_INVALID;
    *layer = nullptr;
    auto *created = new (std::nothrow) fortai_cuda_qwen35_recurrent;
    if (!created) return FORTAI_CUDA_RUNTIME_ERROR;
    created->impl.context = &context->impl;
    created->impl.conv_size = conv_size;
    created->impl.conv_kernel = conv_kernel;
    created->impl.state_size = state_size;
    created->impl.key_heads = key_heads;
    created->impl.value_heads = value_heads;
    created->impl.head_size = head_size;
    created->impl.inner_size = inner_size;
    created->impl.norm_epsilon = norm_epsilon;
    cudaSetDevice(context->impl.device);
    cudaError_t error = allocate_and_copy_float(&created->impl, &created->impl.conv_weights,
        conv_weights, static_cast<size_t>(conv_size) * conv_kernel * sizeof(float));
    if (error == cudaSuccess) error = allocate_and_copy_float(&created->impl, &created->impl.ssm_a,
        ssm_a, static_cast<size_t>(value_heads) * sizeof(float));
    if (error == cudaSuccess) error = allocate_and_copy_float(&created->impl, &created->impl.ssm_dt,
        ssm_dt, static_cast<size_t>(value_heads) * sizeof(float));
    if (error == cudaSuccess) error = allocate_and_copy_float(&created->impl, &created->impl.ssm_norm,
        ssm_norm, static_cast<size_t>(head_size) * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.conv_state),
        static_cast<size_t>(conv_kernel - 1) * conv_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.gdn_state),
        static_cast<size_t>(value_heads) * head_size * head_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.conv_state_first),
        static_cast<size_t>(conv_kernel - 1) * conv_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.gdn_state_first),
        static_cast<size_t>(value_heads) * head_size * head_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.conv_state_second),
        static_cast<size_t>(conv_kernel - 1) * conv_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.gdn_state_second),
        static_cast<size_t>(value_heads) * head_size * head_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.gdn_output),
        static_cast<size_t>(inner_size) * sizeof(float));
    if (error == cudaSuccess) error = cudaMemsetAsync(created->impl.conv_state, 0,
        static_cast<size_t>(conv_kernel - 1) * conv_size * sizeof(float), context->impl.stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(created->impl.gdn_state, 0,
        static_cast<size_t>(value_heads) * head_size * head_size * sizeof(float), context->impl.stream);
    /* The caller fences the complete model setup once all recurrent states
     * have been created.  Keep initialization stream-ordered without a
     * host round-trip for every layer. */
    if (error != cudaSuccess) {
        free_recurrent_device_buffers(&created->impl);
        delete created;
        return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "Qwen recurrent state create", error);
    }
    *layer = created;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_qwen35_recurrent_destroy(fortai_cuda_qwen35_recurrent *layer) {
    if (!layer) return FORTAI_CUDA_OK;
    if (layer->impl.context) {
        cudaSetDevice(layer->impl.context->device);
        free_recurrent_device_buffers(&layer->impl);
    }
    delete layer;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_qwen35_recurrent_reset(fortai_cuda_qwen35_recurrent *layer) {
    if (!layer || !layer->impl.context) return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    cudaSetDevice(context->device);
    cudaError_t error = cudaMemsetAsync(layer->impl.conv_state, 0,
        static_cast<size_t>(layer->impl.conv_kernel - 1) * layer->impl.conv_size * sizeof(float),
        context->stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(layer->impl.gdn_state, 0,
        static_cast<size_t>(layer->impl.value_heads) * layer->impl.head_size * layer->impl.head_size * sizeof(float),
        context->stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->stream);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen recurrent reset", error);
}

extern "C" int fortai_cuda_qwen35_recurrent_run(fortai_cuda_qwen35_recurrent *layer,
    const void *host_activation, size_t activation_bytes, float *host_output,
    size_t output_bytes, float *elapsed_ms) {
    if (!layer || !layer->impl.context || !host_activation || !host_output || !elapsed_ms)
        return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    auto *output_weights = layer->impl.output_weights;
    const size_t expected_activation = static_cast<size_t>(layer->impl.qkv_weights->blocks) * q8_block_bytes;
    const size_t output_activation = static_cast<size_t>(output_weights->blocks) * q8_block_bytes;
    const size_t expected_output = static_cast<size_t>(output_weights->rows) * sizeof(float);
    if (activation_bytes < expected_activation || output_bytes < expected_output)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->device);
    cudaError_t error = ensure_host_matvec_scratch(context,
        std::max(expected_activation, output_activation), expected_output);
    if (error == cudaSuccess) error = cudaEventRecord(context->start, context->stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(context->scratch_raw, host_activation,
        expected_activation, cudaMemcpyHostToDevice, context->stream);
    if (error == cudaSuccess) {
        launch_q8_repack_activation(context->scratch_raw, context->scratch_activation,
            layer->impl.qkv_weights->blocks, context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(layer->impl.qkv_weights, context->scratch_activation, layer->impl.qkv_output,
            context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(layer->impl.gate_weights, context->scratch_activation, layer->impl.gate_output,
            context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(layer->impl.alpha_weights, context->scratch_activation, layer->impl.alpha_output,
            context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(layer->impl.beta_weights, context->scratch_activation, layer->impl.beta_output,
            context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        const int blocks = (layer->impl.conv_size + 255) / 256;
        qwen_recurrent_conv_silu<<<blocks, 256, 0, context->stream>>>(layer->impl.qkv_output,
            layer->impl.conv_weights, layer->impl.conv_state, layer->impl.conv_size,
            layer->impl.conv_kernel);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        const int slices = 2 * layer->impl.key_heads;
        qwen_recurrent_l2_normalize<<<slices, 128, 4 * sizeof(float), context->stream>>>(
            layer->impl.qkv_output, slices, layer->impl.head_size, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_qwen_recurrent_gdn_fused(
            layer->impl.qkv_output, layer->impl.alpha_output,
            layer->impl.beta_output, layer->impl.ssm_a, layer->impl.ssm_dt,
            layer->impl.gdn_state, layer->impl.gdn_output,
            layer->impl.state_size, layer->impl.key_heads, layer->impl.value_heads,
            layer->impl.head_size, context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_recurrent_gdn_norm_gate<<<layer->impl.value_heads, 128,
            4 * sizeof(float), context->stream>>>(
            layer->impl.gdn_output, layer->impl.gate_output, layer->impl.ssm_norm,
            layer->impl.value_heads, layer->impl.head_size, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        const int blocks = output_weights->blocks;
        qwen_quantize_q8<<<(blocks + 31) / 32, 32, 0, context->stream>>>(layer->impl.gdn_output,
            context->scratch_activation, blocks);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(output_weights, context->scratch_activation, context->scratch_output,
            context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaMemcpyAsync(host_output, context->scratch_output,
        expected_output, cudaMemcpyDeviceToHost, context->stream);
    if (error == cudaSuccess) error = cudaEventRecord(context->stop, context->stream);
    if (error == cudaSuccess) error = cudaEventSynchronize(context->stop);
    if (error == cudaSuccess) error = cudaEventElapsedTime(elapsed_ms, context->start, context->stop);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen recurrent run", error);
}

extern "C" int fortai_cuda_qwen35_recurrent_run_device(
    fortai_cuda_qwen35_recurrent *layer, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    if (!layer || !layer->impl.context || !device_activation || !device_output)
        return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    auto *output_weights = layer->impl.output_weights;
    const size_t expected_input = static_cast<size_t>(layer->impl.qkv_weights->blocks) * q8_block_width;
    const size_t expected_output = static_cast<size_t>(output_weights->rows);
    if (activation_elements != expected_input || output_elements < expected_output)
        return FORTAI_CUDA_INVALID;
    const int input_blocks = layer->impl.qkv_weights->blocks;
    const size_t input_bytes = static_cast<size_t>(input_blocks) * q8_block_bytes;
    const size_t output_bytes = static_cast<size_t>(output_weights->blocks) * q8_block_bytes;
    cudaSetDevice(context->device);
    cudaError_t error = ensure_host_matvec_scratch(context,
        std::max(input_bytes, output_bytes),
        static_cast<size_t>(output_weights->rows) * sizeof(float));
    if (error == cudaSuccess) {
        qwen_quantize_q8<<<(input_blocks + 31) / 32, 32, 0, context->stream>>>(
            static_cast<const float *>(device_activation), context->scratch_activation, input_blocks);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(layer->impl.qkv_weights, context->scratch_activation,
            layer->impl.qkv_output, context->stream);
        launch_q8(layer->impl.gate_weights, context->scratch_activation,
            layer->impl.gate_output, context->stream);
        launch_q8(layer->impl.alpha_weights, context->scratch_activation,
            layer->impl.alpha_output, context->stream);
        launch_q8(layer->impl.beta_weights, context->scratch_activation,
            layer->impl.beta_output, context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        const int blocks = (layer->impl.conv_size + 255) / 256;
        qwen_recurrent_conv_silu<<<blocks, 256, 0, context->stream>>>(layer->impl.qkv_output,
            layer->impl.conv_weights, layer->impl.conv_state, layer->impl.conv_size,
            layer->impl.conv_kernel);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        const int slices = 2 * layer->impl.key_heads;
        qwen_recurrent_l2_normalize<<<slices, 128, 4 * sizeof(float), context->stream>>>(
            layer->impl.qkv_output, slices, layer->impl.head_size, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_qwen_recurrent_gdn_fused(
            layer->impl.qkv_output, layer->impl.alpha_output,
            layer->impl.beta_output, layer->impl.ssm_a, layer->impl.ssm_dt,
            layer->impl.gdn_state, layer->impl.gdn_output,
            layer->impl.state_size, layer->impl.key_heads, layer->impl.value_heads,
            layer->impl.head_size, context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_recurrent_gdn_norm_gate<<<layer->impl.value_heads, 128,
            4 * sizeof(float), context->stream>>>(
            layer->impl.gdn_output, layer->impl.gate_output, layer->impl.ssm_norm,
            layer->impl.value_heads, layer->impl.head_size, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_quantize_q8<<<(output_weights->blocks + 31) / 32, 32, 0, context->stream>>>(
            layer->impl.gdn_output, context->scratch_activation, output_weights->blocks);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(output_weights, context->scratch_activation,
            static_cast<float *>(device_output), context->stream);
        error = cudaGetLastError();
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen recurrent device run", error);
}

extern "C" int fortai_cuda_qwen35_recurrent_run_core_device(
    fortai_cuda_qwen35_recurrent *layer, void *device_qkv, size_t qkv_elements,
    const void *device_gate, size_t gate_elements, const void *device_alpha,
    size_t alpha_elements, const void *device_beta, size_t beta_elements,
    void *device_output, size_t output_elements) {
    if (!layer || !layer->impl.context || !device_qkv || !device_gate || !device_alpha ||
        !device_beta || !device_output)
        return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    if (qkv_elements != static_cast<size_t>(layer->impl.conv_size) ||
        gate_elements != static_cast<size_t>(layer->impl.inner_size) ||
        alpha_elements != static_cast<size_t>(layer->impl.value_heads) ||
        beta_elements != static_cast<size_t>(layer->impl.value_heads) ||
        output_elements < static_cast<size_t>(layer->impl.inner_size))
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->device);
    cudaError_t error = cudaSuccess;
    const int blocks = (layer->impl.conv_size + 255) / 256;
    qwen_recurrent_conv_silu<<<blocks, 256, 0, context->stream>>>(
        static_cast<float *>(device_qkv), layer->impl.conv_weights, layer->impl.conv_state,
        layer->impl.conv_size, layer->impl.conv_kernel);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
        const int slices = 2 * layer->impl.key_heads;
        qwen_recurrent_l2_normalize<<<slices, 128, 4 * sizeof(float), context->stream>>>(
            static_cast<float *>(device_qkv), slices, layer->impl.head_size, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_qwen_recurrent_gdn_fused(static_cast<const float *>(device_qkv),
            static_cast<const float *>(device_alpha),
            static_cast<const float *>(device_beta), layer->impl.ssm_a, layer->impl.ssm_dt,
            layer->impl.gdn_state, layer->impl.gdn_output, layer->impl.state_size,
            layer->impl.key_heads, layer->impl.value_heads, layer->impl.head_size,
            context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_recurrent_gdn_norm_gate<<<layer->impl.value_heads, 128,
            4 * sizeof(float), context->stream>>>(
            layer->impl.gdn_output, static_cast<const float *>(device_gate), layer->impl.ssm_norm,
            layer->impl.value_heads, layer->impl.head_size, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(device_output, layer->impl.gdn_output,
            static_cast<size_t>(layer->impl.inner_size) * sizeof(float),
            cudaMemcpyDeviceToDevice, context->stream);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen recurrent core device", error);
}

extern "C" int fortai_cuda_qwen35_recurrent_run_core_device_batch(
    fortai_cuda_qwen35_recurrent *layer, void *device_qkv, size_t qkv_elements,
    const void *device_gate, size_t gate_elements, const void *device_alpha,
    size_t alpha_elements, const void *device_beta, size_t beta_elements,
    int batch, void *device_output, size_t output_elements) {
    if (!layer || !layer->impl.context || !device_qkv || !device_gate ||
        !device_alpha || !device_beta || !device_output || batch <= 0 ||
        batch > INT32_MAX) return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    const size_t qkv_stride = static_cast<size_t>(layer->impl.conv_size);
    const size_t gate_stride = static_cast<size_t>(layer->impl.inner_size);
    const size_t alpha_stride = static_cast<size_t>(layer->impl.value_heads);
    if (qkv_elements < qkv_stride * batch || gate_elements < gate_stride * batch ||
        alpha_elements < alpha_stride * batch || beta_elements < alpha_stride * batch ||
        output_elements < gate_stride * batch) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->device);
    cudaError_t error = cudaSuccess;
    qwen_recurrent_conv_silu_batch<<<(layer->impl.conv_size + 255) / 256, 256, 0,
        context->stream>>>(static_cast<float *>(device_qkv), layer->impl.conv_weights,
        layer->impl.conv_state, layer->impl.conv_state_first, layer->impl.conv_state_second,
        layer->impl.conv_size, layer->impl.conv_kernel, batch);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
        const int slices = 2 * layer->impl.key_heads;
        qwen_recurrent_l2_normalize_batch<<<dim3(static_cast<unsigned>(slices),
            static_cast<unsigned>(batch), 1), 128, 0, context->stream>>>(
            static_cast<float *>(device_qkv), slices, layer->impl.head_size, batch,
            layer->impl.conv_size, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        if (layer->impl.state_size == 128 && layer->impl.head_size == 128) {
            qwen_recurrent_gdn_fused_batch_fixed<128><<<dim3(
                static_cast<unsigned>(layer->impl.value_heads), 1, 128 / 4),
                dim3(32, 4, 1), 0, context->stream>>>(
                static_cast<const float *>(device_qkv), static_cast<const float *>(device_alpha),
                static_cast<const float *>(device_beta), layer->impl.ssm_a, layer->impl.ssm_dt,
                layer->impl.gdn_state, layer->impl.gdn_state_first, layer->impl.gdn_state_second,
                static_cast<float *>(device_output), layer->impl.state_size, layer->impl.key_heads,
                layer->impl.value_heads, layer->impl.inner_size, batch);
        } else {
            qwen_recurrent_gdn_fused_batch_generic<<<
                (layer->impl.value_heads * layer->impl.head_size + 3) / 4,
                dim3(32, 4, 1), 0, context->stream>>>(static_cast<const float *>(device_qkv),
                static_cast<const float *>(device_alpha), static_cast<const float *>(device_beta),
                layer->impl.ssm_a, layer->impl.ssm_dt, layer->impl.gdn_state,
                layer->impl.gdn_state_first, layer->impl.gdn_state_second,
                static_cast<float *>(device_output), layer->impl.state_size,
                layer->impl.key_heads, layer->impl.value_heads, layer->impl.head_size,
                layer->impl.inner_size, batch);
        }
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_recurrent_gdn_norm_gate_batch<<<dim3(
            static_cast<unsigned>(layer->impl.value_heads),
            static_cast<unsigned>(batch), 1), 128, 0, context->stream>>>(
            static_cast<float *>(device_output), static_cast<const float *>(device_gate),
            layer->impl.ssm_norm, layer->impl.value_heads, layer->impl.head_size,
            layer->impl.inner_size, batch, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen recurrent batched core", error);
}

extern "C" int fortai_cuda_qwen35_recurrent_restore_first(
    fortai_cuda_qwen35_recurrent *layer) {
    if (!layer || !layer->impl.context || !layer->impl.conv_state ||
        !layer->impl.gdn_state || !layer->impl.conv_state_first ||
        !layer->impl.gdn_state_first) return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    cudaSetDevice(context->device);
    const size_t conv_bytes = static_cast<size_t>(layer->impl.conv_kernel - 1) *
        layer->impl.conv_size * sizeof(float);
    const size_t gdn_bytes = static_cast<size_t>(layer->impl.value_heads) *
        layer->impl.head_size * layer->impl.head_size * sizeof(float);
    cudaError_t error = cudaMemcpyAsync(layer->impl.conv_state,
        layer->impl.conv_state_first, conv_bytes, cudaMemcpyDeviceToDevice, context->stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(layer->impl.gdn_state,
        layer->impl.gdn_state_first, gdn_bytes, cudaMemcpyDeviceToDevice, context->stream);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen recurrent rollback", error);
}

extern "C" int fortai_cuda_qwen35_recurrent_restore_second(
    fortai_cuda_qwen35_recurrent *layer) {
    if (!layer || !layer->impl.context || !layer->impl.conv_state ||
        !layer->impl.gdn_state || !layer->impl.conv_state_second ||
        !layer->impl.gdn_state_second) return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    cudaSetDevice(context->device);
    const size_t conv_bytes = static_cast<size_t>(layer->impl.conv_kernel - 1) *
        layer->impl.conv_size * sizeof(float);
    const size_t gdn_bytes = static_cast<size_t>(layer->impl.value_heads) *
        layer->impl.head_size * layer->impl.head_size * sizeof(float);
    cudaError_t error = cudaMemcpyAsync(layer->impl.conv_state,
        layer->impl.conv_state_second, conv_bytes, cudaMemcpyDeviceToDevice, context->stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(layer->impl.gdn_state,
        layer->impl.gdn_state_second, gdn_bytes, cudaMemcpyDeviceToDevice, context->stream);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen recurrent second-row rollback", error);
}

extern "C" int fortai_cuda_qwen35_attention_create(
    fortai_cuda_q8_context *context, const fortai_cuda_q8_weights *query_weights,
    const fortai_cuda_q8_weights *key_weights, const fortai_cuda_q8_weights *value_weights,
    const fortai_cuda_q8_weights *output_weights, const void *query_norm,
    size_t query_norm_bytes, const void *key_norm, size_t key_norm_bytes,
    int heads, int key_value_heads, int head_size, int value_size, int max_context,
    int rope_dimension, float rope_base, float norm_epsilon,
    int cache_key_type, int cache_value_type,
    fortai_cuda_qwen35_attention **layer) {
    if (!context || !layer || !query_weights || !key_weights || !value_weights ||
        !output_weights || !query_norm || !key_norm || heads <= 0 || key_value_heads <= 0 ||
        head_size <= 0 || value_size <= 0 || max_context <= 0 ||
        heads % key_value_heads != 0 || rope_dimension < 0 ||
        (rope_dimension > 0 && (rope_dimension > head_size || rope_dimension % 2 != 0)) ||
        rope_base <= 0.0f || norm_epsilon <= 0.0f ||
        cache_key_type < 0 || cache_key_type > 2 || cache_value_type < 0 || cache_value_type > 2 ||
        (cache_key_type != 0 && head_size % q8_block_width != 0) ||
        (cache_value_type != 0 && value_size % q8_block_width != 0))
        return FORTAI_CUDA_INVALID;
    auto *query = const_cast<fortai_cuda_q8_weights_impl *>(&query_weights->impl);
    auto *key = const_cast<fortai_cuda_q8_weights_impl *>(&key_weights->impl);
    auto *value = const_cast<fortai_cuda_q8_weights_impl *>(&value_weights->impl);
    auto *output = const_cast<fortai_cuda_q8_weights_impl *>(&output_weights->impl);
    if (query->context != &context->impl || key->context != &context->impl ||
        value->context != &context->impl || output->context != &context->impl ||
        query->rows != heads * 2 * head_size || key->rows != key_value_heads * head_size ||
        value->rows != key_value_heads * value_size ||
        output->blocks * q8_block_width != heads * value_size ||
        query_norm_bytes < static_cast<size_t>(head_size) * sizeof(float) ||
        key_norm_bytes < static_cast<size_t>(head_size) * sizeof(float))
        return FORTAI_CUDA_INVALID;
    *layer = nullptr;
    auto *created = new (std::nothrow) fortai_cuda_qwen35_attention;
    if (!created) return FORTAI_CUDA_RUNTIME_ERROR;
    created->impl.context = &context->impl;
    created->impl.query_weights = query;
    created->impl.key_weights = key;
    created->impl.value_weights = value;
    created->impl.output_weights = output;
    created->impl.heads = heads;
    created->impl.key_value_heads = key_value_heads;
    created->impl.head_size = head_size;
    created->impl.value_size = value_size;
    created->impl.max_context = max_context;
    created->impl.rope_dimension = rope_dimension;
    created->impl.rope_base = rope_base;
    created->impl.norm_epsilon = norm_epsilon;
    created->impl.cache_key_type = cache_key_type;
    created->impl.cache_value_type = cache_value_type;
    cudaSetDevice(context->impl.device);
    cudaError_t error = allocate_and_copy_float_attention(&created->impl,
        &created->impl.query_norm, query_norm, static_cast<size_t>(head_size) * sizeof(float));
    if (error == cudaSuccess) error = allocate_and_copy_float_attention(&created->impl,
        &created->impl.key_norm, key_norm, static_cast<size_t>(head_size) * sizeof(float));
    if (error == cudaSuccess) {
        if (created->impl.cache_key_type != 0) {
            const size_t bytes = static_cast<size_t>(max_context) * key_value_heads *
                attention_cache_row_bytes(created->impl.cache_key_type, head_size);
            error = cudaMalloc(reinterpret_cast<void **>(&created->impl.key_cache_q8), bytes);
        } else {
            error = cudaMalloc(reinterpret_cast<void **>(&created->impl.key_cache),
                static_cast<size_t>(max_context) * key_value_heads * head_size * sizeof(__half));
        }
    }
    if (error == cudaSuccess) {
        if (created->impl.cache_value_type != 0) {
            const size_t bytes = static_cast<size_t>(max_context) * key_value_heads *
                attention_cache_row_bytes(created->impl.cache_value_type, value_size);
            error = cudaMalloc(reinterpret_cast<void **>(&created->impl.value_cache_q8), bytes);
        } else {
            error = cudaMalloc(reinterpret_cast<void **>(&created->impl.value_cache),
                static_cast<size_t>(max_context) * key_value_heads * value_size * sizeof(__half));
        }
    }
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.query),
        static_cast<size_t>(heads) * 2 * head_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.key),
        static_cast<size_t>(key_value_heads) * head_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.value),
        static_cast<size_t>(key_value_heads) * value_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.attention),
        static_cast<size_t>(heads) * value_size * sizeof(float));
    if (error == cudaSuccess) {
        const size_t bytes = static_cast<size_t>(max_context) * key_value_heads *
            attention_cache_row_bytes(created->impl.cache_key_type, head_size);
        void *cache = created->impl.cache_key_type != 0 ?
            static_cast<void *>(created->impl.key_cache_q8) :
            static_cast<void *>(created->impl.key_cache);
        error = cudaMemsetAsync(cache, 0, bytes, context->impl.stream);
    }
    if (error == cudaSuccess) {
        const size_t bytes = static_cast<size_t>(max_context) * key_value_heads *
            attention_cache_row_bytes(created->impl.cache_value_type, value_size);
        void *cache = created->impl.cache_value_type != 0 ?
            static_cast<void *>(created->impl.value_cache_q8) :
            static_cast<void *>(created->impl.value_cache);
        error = cudaMemsetAsync(cache, 0, bytes, context->impl.stream);
    }
    /* Attention state uploads and cache clears are ordered on this context's
     * stream; setup performs the single completion fence after all layers. */
    if (error != cudaSuccess) {
        free_attention_device_buffers(&created->impl);
        delete created;
        return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "Qwen attention create", error);
    }
    *layer = created;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_qwen35_attention_create_state(
    fortai_cuda_q8_context *context, const void *query_norm, size_t query_norm_bytes,
    const void *key_norm, size_t key_norm_bytes, int heads, int key_value_heads,
    int head_size, int value_size, int max_context, int rope_dimension,
    float rope_base, float norm_epsilon, int cache_key_type, int cache_value_type,
    fortai_cuda_qwen35_attention **layer) {
    if (!context || !layer || !query_norm || !key_norm || heads <= 0 ||
        key_value_heads <= 0 || head_size <= 0 || value_size <= 0 || max_context <= 0 ||
        heads % key_value_heads != 0 || rope_dimension < 0 ||
        (rope_dimension > 0 && (rope_dimension > head_size || rope_dimension % 2 != 0)) ||
        rope_base <= 0.0f || norm_epsilon <= 0.0f ||
        cache_key_type < 0 || cache_key_type > 2 || cache_value_type < 0 || cache_value_type > 2 ||
        (cache_key_type != 0 && head_size % q8_block_width != 0) ||
        (cache_value_type != 0 && value_size % q8_block_width != 0) ||
        query_norm_bytes < static_cast<size_t>(head_size) * sizeof(float) ||
        key_norm_bytes < static_cast<size_t>(head_size) * sizeof(float))
        return FORTAI_CUDA_INVALID;
    *layer = nullptr;
    auto *created = new (std::nothrow) fortai_cuda_qwen35_attention;
    if (!created) return FORTAI_CUDA_RUNTIME_ERROR;
    created->impl.context = &context->impl;
    created->impl.heads = heads;
    created->impl.key_value_heads = key_value_heads;
    created->impl.head_size = head_size;
    created->impl.value_size = value_size;
    created->impl.max_context = max_context;
    created->impl.rope_dimension = rope_dimension;
    created->impl.rope_base = rope_base;
    created->impl.norm_epsilon = norm_epsilon;
    created->impl.cache_key_type = cache_key_type;
    created->impl.cache_value_type = cache_value_type;
    cudaSetDevice(context->impl.device);
    cudaError_t error = allocate_and_copy_float_attention(&created->impl,
        &created->impl.query_norm, query_norm, static_cast<size_t>(head_size) * sizeof(float));
    if (error == cudaSuccess) error = allocate_and_copy_float_attention(&created->impl,
        &created->impl.key_norm, key_norm, static_cast<size_t>(head_size) * sizeof(float));
    if (error == cudaSuccess) {
        if (created->impl.cache_key_type != 0) {
            const size_t bytes = static_cast<size_t>(max_context) * key_value_heads *
                attention_cache_row_bytes(created->impl.cache_key_type, head_size);
            error = cudaMalloc(reinterpret_cast<void **>(&created->impl.key_cache_q8), bytes);
        } else {
            error = cudaMalloc(reinterpret_cast<void **>(&created->impl.key_cache),
                static_cast<size_t>(max_context) * key_value_heads * head_size * sizeof(__half));
        }
    }
    if (error == cudaSuccess) {
        if (created->impl.cache_value_type != 0) {
            const size_t bytes = static_cast<size_t>(max_context) * key_value_heads *
                attention_cache_row_bytes(created->impl.cache_value_type, value_size);
            error = cudaMalloc(reinterpret_cast<void **>(&created->impl.value_cache_q8), bytes);
        } else {
            error = cudaMalloc(reinterpret_cast<void **>(&created->impl.value_cache),
                static_cast<size_t>(max_context) * key_value_heads * value_size * sizeof(__half));
        }
    }
    if (error == cudaSuccess) {
        const size_t bytes = static_cast<size_t>(max_context) * key_value_heads *
            attention_cache_row_bytes(created->impl.cache_key_type, head_size);
        void *cache = created->impl.cache_key_type != 0 ?
            static_cast<void *>(created->impl.key_cache_q8) :
            static_cast<void *>(created->impl.key_cache);
        error = cudaMemsetAsync(cache, 0, bytes, context->impl.stream);
    }
    if (error == cudaSuccess) {
        const size_t bytes = static_cast<size_t>(max_context) * key_value_heads *
            attention_cache_row_bytes(created->impl.cache_value_type, value_size);
        void *cache = created->impl.cache_value_type != 0 ?
            static_cast<void *>(created->impl.value_cache_q8) :
            static_cast<void *>(created->impl.value_cache);
        error = cudaMemsetAsync(cache, 0, bytes, context->impl.stream);
    }
    /* Do not fence each attention state during model construction.  All
     * subsequent operations use the same stream and setup synchronizes it
     * once after the complete resident graph is assembled. */
    if (error != cudaSuccess) {
        free_attention_device_buffers(&created->impl);
        delete created;
        return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "Qwen attention state create", error);
    }
    *layer = created;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_qwen35_attention_destroy(fortai_cuda_qwen35_attention *layer) {
    if (!layer) return FORTAI_CUDA_OK;
    if (layer->impl.context) {
        cudaSetDevice(layer->impl.context->device);
        free_attention_device_buffers(&layer->impl);
    }
    delete layer;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_qwen35_attention_reset(fortai_cuda_qwen35_attention *layer) {
    if (!layer || !layer->impl.context) return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    cudaSetDevice(context->device);
    const size_t key_bytes = static_cast<size_t>(layer->impl.max_context) *
        layer->impl.key_value_heads *
        attention_cache_row_bytes(layer->impl.cache_key_type, layer->impl.head_size);
    void *key_cache = layer->impl.cache_key_type != 0 ?
        static_cast<void *>(layer->impl.key_cache_q8) :
        static_cast<void *>(layer->impl.key_cache);
    const size_t value_bytes = static_cast<size_t>(layer->impl.max_context) *
        layer->impl.key_value_heads *
        attention_cache_row_bytes(layer->impl.cache_value_type, layer->impl.value_size);
    void *value_cache = layer->impl.cache_value_type != 0 ?
        static_cast<void *>(layer->impl.value_cache_q8) :
        static_cast<void *>(layer->impl.value_cache);
    cudaError_t error = cudaMemsetAsync(key_cache, 0, key_bytes, context->stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(value_cache, 0, value_bytes, context->stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->stream);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen attention reset", error);
}

extern "C" int fortai_cuda_qwen35_attention_run_device(
    fortai_cuda_qwen35_attention *layer, const void *device_activation,
    size_t activation_elements, int position, void *device_output, size_t output_elements) {
    if (!layer || !layer->impl.context || !device_activation || !device_output ||
        position < 0 || position >= layer->impl.max_context)
        return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    auto *query = layer->impl.query_weights;
    auto *key = layer->impl.key_weights;
    auto *value = layer->impl.value_weights;
    auto *output = layer->impl.output_weights;
    const size_t expected_input = static_cast<size_t>(query->blocks) * q8_block_width;
    const size_t attention_elements = static_cast<size_t>(layer->impl.heads) * layer->impl.value_size;
    if (activation_elements != expected_input || output_elements < static_cast<size_t>(output->rows) ||
        query->blocks != key->blocks || query->blocks != value->blocks ||
        output->blocks * q8_block_width != static_cast<int>(attention_elements))
        return FORTAI_CUDA_INVALID;
    const size_t input_bytes = static_cast<size_t>(query->blocks) * q8_block_bytes;
    const size_t output_bytes = static_cast<size_t>(output->blocks) * q8_block_bytes;
    cudaSetDevice(context->device);
    cudaError_t error = ensure_host_matvec_scratch(context,
        std::max(input_bytes, output_bytes),
        static_cast<size_t>(output->rows) * sizeof(float));
    if (error == cudaSuccess) {
        qwen_quantize_q8<<<(query->blocks + 31) / 32, 32, 0, context->stream>>>(
            static_cast<const float *>(device_activation), context->scratch_activation,
            query->blocks);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        const host_matvec_request requests[] = {
            {query, layer->impl.query, static_cast<size_t>(query->rows) * sizeof(float)},
            {key, layer->impl.key, static_cast<size_t>(key->rows) * sizeof(float)},
            {value, layer->impl.value, static_cast<size_t>(value->rows) * sizeof(float)}};
        launch_q8_grouped(requests, 3, context->scratch_activation, context->stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_attention_prepare<<<layer->impl.heads, 256, 512 * sizeof(float), context->stream>>>(
            layer->impl.query, layer->impl.key, layer->impl.value, layer->impl.query_norm,
            layer->impl.key_norm, layer->impl.key_cache, layer->impl.value_cache,
            layer->impl.key_cache_q8, layer->impl.value_cache_q8,
            layer->impl.cache_key_type, layer->impl.cache_value_type,
            layer->impl.heads, layer->impl.key_value_heads, layer->impl.head_size,
            layer->impl.value_size, layer->impl.max_context, context->position,
            layer->impl.rope_dimension, layer->impl.rope_base, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        if (layer->impl.cache_key_type == 0 && layer->impl.cache_value_type == 0 &&
            layer->impl.max_context <= 12000) {
            const char *enable_vector_env = std::getenv("FORTAI_CUDA_ATTENTION_VECTOR");
            const bool enable_vector = enable_vector_env && enable_vector_env[0] == '1';
            const char *enable_exact_vector_env = std::getenv("FORTAI_CUDA_ATTENTION_EXACT_VECTOR");
            const bool enable_exact_vector = enable_exact_vector_env && enable_exact_vector_env[0] == '1';
            const char *enable_scalar_mma_env = std::getenv("FORTAI_CUDA_ATTENTION_MMA_SCALAR");
            const bool enable_scalar_mma = enable_scalar_mma_env && enable_scalar_mma_env[0] == '1';
            if (enable_scalar_mma && layer->impl.head_size == 256 && layer->impl.value_size == 256) {
                const int query_stride = 2 * layer->impl.head_size * layer->impl.heads;
                const int attention_stride = layer->impl.value_size * layer->impl.heads;
                const size_t shared_bytes =
                    (static_cast<size_t>(16 * layer->impl.head_size + 64 * 16 +
                        16 * 64 + 8 * 16 * 16) * sizeof(__half)) +
                    static_cast<size_t>(16 * 64 + 16 * 3 + 8 * 16 * 16 +
                        16 * layer->impl.value_size) * sizeof(float);
                error = cudaFuncSetAttribute(qwen_attention_apply_mma_f16_batch,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
                if (error == cudaSuccess)
                    qwen_attention_apply_mma_f16_batch<<<dim3(
                        static_cast<unsigned>(layer->impl.heads), 1, 1), 256,
                        shared_bytes, context->stream>>>(layer->impl.query,
                        layer->impl.key_cache, layer->impl.value_cache,
                        layer->impl.attention, layer->impl.heads, layer->impl.key_value_heads,
                        layer->impl.head_size, layer->impl.value_size, layer->impl.max_context,
                        context->position_value, query_stride,
                        attention_stride, 1, 2 * layer->impl.head_size);
            } else if (enable_exact_vector && layer->impl.head_size == 256 &&
                layer->impl.value_size == 256) {
                qwen_attention_apply_f16_exact_vector<<<static_cast<unsigned>(layer->impl.heads),
                    dim3(32, 4, 1),
                    static_cast<size_t>(128 + 8 + 4 * layer->impl.value_size) * sizeof(float),
                    context->stream>>>(
                    layer->impl.query, layer->impl.key_cache, layer->impl.value_cache,
                    layer->impl.attention, layer->impl.heads, layer->impl.key_value_heads,
                    layer->impl.head_size, layer->impl.value_size, layer->impl.max_context,
                    context->position);
            } else if (enable_vector && layer->impl.head_size == 256 && layer->impl.value_size == 256) {
                qwen_attention_apply_f16_vector<<<static_cast<unsigned>(layer->impl.heads),
                    dim3(32, 4, 1), 0, context->stream>>>(
                    layer->impl.query, layer->impl.key_cache, layer->impl.value_cache,
                    layer->impl.attention, layer->impl.heads, layer->impl.key_value_heads,
                    layer->impl.head_size, layer->impl.value_size, layer->impl.max_context,
                    context->position);
            } else {
                const char *enable_gqa = std::getenv("FORTAI_CUDA_ATTENTION_GQA_SCALAR");
                const bool use_gqa = enable_gqa && enable_gqa[0] == '1' &&
                    (layer->impl.head_size == 128 || layer->impl.head_size == 256) &&
                    layer->impl.value_size == layer->impl.head_size &&
                    layer->impl.heads == 4 * layer->impl.key_value_heads;
                if (use_gqa) {
                if (layer->impl.head_size == 128) {
                    constexpr int gqa_key_tile = 128;
                    constexpr size_t gqa_shared_bytes =
                        static_cast<size_t>(2 * gqa_key_tile * 128) * sizeof(__half);
                    error = cudaFuncSetAttribute(qwen_attention_apply_gqa_f16_batch<4, 128, 128, 1, gqa_key_tile>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(gqa_shared_bytes));
                    if (error == cudaSuccess)
                        qwen_attention_apply_gqa_f16_batch<4, 128, 128, 1, gqa_key_tile><<<dim3(
                            static_cast<unsigned>(layer->impl.key_value_heads), 1, 1), 128,
                            gqa_shared_bytes, context->stream>>>(
                            layer->impl.query, layer->impl.key_cache, layer->impl.value_cache,
                            layer->impl.attention, layer->impl.heads, layer->impl.key_value_heads,
                            layer->impl.max_context, 0,
                            2 * layer->impl.heads * layer->impl.head_size,
                            layer->impl.heads * layer->impl.value_size, 1, context->position);
                } else {
                    constexpr int gqa_key_tile = 64;
                    constexpr size_t gqa_shared_bytes =
                        static_cast<size_t>(2 * gqa_key_tile * 256) * sizeof(__half);
                    error = cudaFuncSetAttribute(qwen_attention_apply_gqa_f16_batch<4, 256, 256, 1, gqa_key_tile>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(gqa_shared_bytes));
                    if (error == cudaSuccess)
                        qwen_attention_apply_gqa_f16_batch<4, 256, 256, 1, gqa_key_tile><<<dim3(
                            static_cast<unsigned>(layer->impl.key_value_heads), 1, 1), 128,
                            gqa_shared_bytes, context->stream>>>(
                            layer->impl.query, layer->impl.key_cache, layer->impl.value_cache,
                            layer->impl.attention, layer->impl.heads, layer->impl.key_value_heads,
                            layer->impl.max_context, 0,
                            2 * layer->impl.heads * layer->impl.head_size,
                            layer->impl.heads * layer->impl.value_size, 1, context->position);
                }
                } else {
                qwen_attention_apply_f16_llama<<<layer->impl.heads, 128,
                    static_cast<size_t>(128 + 8 + 4 * layer->impl.value_size) * sizeof(float), context->stream>>>(
                    layer->impl.query, layer->impl.key_cache, layer->impl.value_cache,
                    layer->impl.attention, layer->impl.heads, layer->impl.key_value_heads,
                    layer->impl.head_size, layer->impl.value_size, layer->impl.max_context,
                    context->position);
                }
            }
        } else {
            if (qwen_attention_q8_vector_requested() && layer->impl.cache_key_type == 1 &&
                layer->impl.cache_value_type == 1 && layer->impl.head_size == 256 &&
                layer->impl.value_size == 256) {
                qwen_attention_apply_q8_vector<<<static_cast<unsigned>(layer->impl.heads),
                    dim3(32, 4, 1), 0, context->stream>>>(
                    layer->impl.query, layer->impl.key_cache_q8, layer->impl.value_cache_q8,
                    layer->impl.attention, layer->impl.heads, layer->impl.key_value_heads,
                    layer->impl.head_size, layer->impl.value_size, layer->impl.max_context,
                    context->position);
            } else {
                qwen_attention_apply_online<<<layer->impl.heads, 256, 0, context->stream>>>(
                    layer->impl.query, layer->impl.key_cache, layer->impl.key_cache_q8,
                    layer->impl.value_cache, layer->impl.value_cache_q8, layer->impl.attention,
                    layer->impl.heads, layer->impl.key_value_heads, layer->impl.head_size,
                    layer->impl.value_size, layer->impl.max_context, layer->impl.cache_key_type,
                    layer->impl.cache_value_type, context->position);
            }
        }
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_quantize_q8<<<(output->blocks + 31) / 32, 32, 0, context->stream>>>(
            layer->impl.attention, context->scratch_activation, output->blocks);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        launch_q8(output, context->scratch_activation, static_cast<float *>(device_output),
            context->stream);
        error = cudaGetLastError();
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen attention device run", error);
}

extern "C" int fortai_cuda_qwen35_attention_run_core_device(
    fortai_cuda_qwen35_attention *layer, const void *device_query,
    size_t query_elements, const void *device_key, size_t key_elements,
    const void *device_value, size_t value_elements, int position,
    void *device_output, size_t output_elements) {
    if (!layer || !layer->impl.context || !device_query || !device_key || !device_value ||
        !device_output || position < 0 || position >= layer->impl.max_context)
        return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    const size_t expected_query = static_cast<size_t>(layer->impl.heads) * 2 * layer->impl.head_size;
    const size_t expected_key = static_cast<size_t>(layer->impl.key_value_heads) * layer->impl.head_size;
    const size_t expected_value = static_cast<size_t>(layer->impl.key_value_heads) * layer->impl.value_size;
    const size_t expected_output = static_cast<size_t>(layer->impl.heads) * layer->impl.value_size;
    if (query_elements != expected_query || key_elements != expected_key ||
        value_elements != expected_value || output_elements < expected_output)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->device);
    qwen_attention_prepare<<<layer->impl.heads, 256, 512 * sizeof(float), context->stream>>>(
        const_cast<float *>(static_cast<const float *>(device_query)),
        const_cast<float *>(static_cast<const float *>(device_key)),
        static_cast<const float *>(device_value), layer->impl.query_norm, layer->impl.key_norm,
        layer->impl.key_cache, layer->impl.value_cache, layer->impl.key_cache_q8,
        layer->impl.value_cache_q8, layer->impl.cache_key_type, layer->impl.cache_value_type,
        layer->impl.heads, layer->impl.key_value_heads, layer->impl.head_size, layer->impl.value_size,
        layer->impl.max_context, context->position, layer->impl.rope_dimension,
        layer->impl.rope_base, layer->impl.norm_epsilon);
    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        if (layer->impl.cache_key_type == 0 && layer->impl.cache_value_type == 0 &&
            layer->impl.max_context <= 12000) {
            const char *enable_vector_env = std::getenv("FORTAI_CUDA_ATTENTION_VECTOR");
            const bool enable_vector = enable_vector_env && enable_vector_env[0] == '1';
            const char *enable_exact_vector_env = std::getenv("FORTAI_CUDA_ATTENTION_EXACT_VECTOR");
            const bool enable_exact_vector = enable_exact_vector_env && enable_exact_vector_env[0] == '1';
            const char *enable_scalar_mma_env = std::getenv("FORTAI_CUDA_ATTENTION_MMA_SCALAR");
            const bool enable_scalar_mma = enable_scalar_mma_env && enable_scalar_mma_env[0] == '1';
            if (enable_scalar_mma && layer->impl.head_size == 256 && layer->impl.value_size == 256) {
                const int query_stride = 2 * layer->impl.head_size * layer->impl.heads;
                const int attention_stride = layer->impl.value_size * layer->impl.heads;
                const size_t shared_bytes =
                    (static_cast<size_t>(16 * layer->impl.head_size + 64 * 16 +
                        16 * 64 + 8 * 16 * 16) * sizeof(__half)) +
                    static_cast<size_t>(16 * 64 + 16 * 3 + 8 * 16 * 16 +
                        16 * layer->impl.value_size) * sizeof(float);
                error = cudaFuncSetAttribute(qwen_attention_apply_mma_f16_batch,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
                if (error == cudaSuccess)
                    qwen_attention_apply_mma_f16_batch<<<dim3(
                        static_cast<unsigned>(layer->impl.heads), 1, 1), 256,
                        shared_bytes, context->stream>>>(static_cast<const float *>(device_query),
                        layer->impl.key_cache, layer->impl.value_cache,
                        static_cast<float *>(device_output), layer->impl.heads,
                        layer->impl.key_value_heads, layer->impl.head_size,
                        layer->impl.value_size, layer->impl.max_context,
                        position, query_stride,
                        attention_stride, 1, 2 * layer->impl.head_size);
            } else if (enable_exact_vector && layer->impl.head_size == 256 &&
                layer->impl.value_size == 256) {
                qwen_attention_apply_f16_exact_vector<<<static_cast<unsigned>(layer->impl.heads),
                    dim3(32, 4, 1),
                    static_cast<size_t>(128 + 8 + 4 * layer->impl.value_size) * sizeof(float),
                    context->stream>>>(
                    static_cast<const float *>(device_query), layer->impl.key_cache,
                    layer->impl.value_cache, static_cast<float *>(device_output),
                    layer->impl.heads, layer->impl.key_value_heads, layer->impl.head_size,
                    layer->impl.value_size, layer->impl.max_context, context->position);
            } else if (enable_vector && layer->impl.head_size == 256 && layer->impl.value_size == 256) {
                qwen_attention_apply_f16_vector<<<static_cast<unsigned>(layer->impl.heads),
                    dim3(32, 4, 1), 0, context->stream>>>(
                    static_cast<const float *>(device_query), layer->impl.key_cache,
                    layer->impl.value_cache, static_cast<float *>(device_output),
                    layer->impl.heads, layer->impl.key_value_heads, layer->impl.head_size,
                    layer->impl.value_size, layer->impl.max_context, context->position);
            } else {
                const char *enable_gqa = std::getenv("FORTAI_CUDA_ATTENTION_GQA_SCALAR");
                const bool use_gqa = enable_gqa && enable_gqa[0] == '1' &&
                    (layer->impl.head_size == 128 || layer->impl.head_size == 256) &&
                    layer->impl.value_size == layer->impl.head_size &&
                    layer->impl.heads == 4 * layer->impl.key_value_heads;
                if (use_gqa) {
                if (layer->impl.head_size == 128) {
                    constexpr int gqa_key_tile = 128;
                    constexpr size_t gqa_shared_bytes =
                        static_cast<size_t>(2 * gqa_key_tile * 128) * sizeof(__half);
                    error = cudaFuncSetAttribute(qwen_attention_apply_gqa_f16_batch<4, 128, 128, 1, gqa_key_tile>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(gqa_shared_bytes));
                    if (error == cudaSuccess)
                        qwen_attention_apply_gqa_f16_batch<4, 128, 128, 1, gqa_key_tile><<<dim3(
                            static_cast<unsigned>(layer->impl.key_value_heads), 1, 1), 128,
                            gqa_shared_bytes, context->stream>>>(
                            static_cast<const float *>(device_query), layer->impl.key_cache,
                            layer->impl.value_cache, static_cast<float *>(device_output), layer->impl.heads,
                            layer->impl.key_value_heads, layer->impl.max_context, 0,
                            2 * layer->impl.heads * layer->impl.head_size,
                            layer->impl.heads * layer->impl.value_size, 1, context->position);
                } else {
                    constexpr int gqa_key_tile = 64;
                    constexpr size_t gqa_shared_bytes =
                        static_cast<size_t>(2 * gqa_key_tile * 256) * sizeof(__half);
                    error = cudaFuncSetAttribute(qwen_attention_apply_gqa_f16_batch<4, 256, 256, 1, gqa_key_tile>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(gqa_shared_bytes));
                    if (error == cudaSuccess)
                        qwen_attention_apply_gqa_f16_batch<4, 256, 256, 1, gqa_key_tile><<<dim3(
                            static_cast<unsigned>(layer->impl.key_value_heads), 1, 1), 128,
                            gqa_shared_bytes, context->stream>>>(
                            static_cast<const float *>(device_query), layer->impl.key_cache,
                            layer->impl.value_cache, static_cast<float *>(device_output), layer->impl.heads,
                            layer->impl.key_value_heads, layer->impl.max_context, 0,
                            2 * layer->impl.heads * layer->impl.head_size,
                            layer->impl.heads * layer->impl.value_size, 1, context->position);
                }
                } else {
                qwen_attention_apply_f16_llama<<<layer->impl.heads, 128,
                    static_cast<size_t>(128 + 8 + 4 * layer->impl.value_size) * sizeof(float), context->stream>>>(
                    static_cast<const float *>(device_query), layer->impl.key_cache,
                    layer->impl.value_cache, static_cast<float *>(device_output), layer->impl.heads,
                    layer->impl.key_value_heads, layer->impl.head_size, layer->impl.value_size,
                    layer->impl.max_context, context->position);
                }
            }
        } else {
            if (qwen_attention_q8_vector_requested() && layer->impl.cache_key_type == 1 &&
                layer->impl.cache_value_type == 1 && layer->impl.head_size == 256 &&
                layer->impl.value_size == 256) {
                qwen_attention_apply_q8_vector<<<static_cast<unsigned>(layer->impl.heads),
                    dim3(32, 4, 1), 0, context->stream>>>(
                    static_cast<const float *>(device_query), layer->impl.key_cache_q8,
                    layer->impl.value_cache_q8, static_cast<float *>(device_output),
                    layer->impl.heads, layer->impl.key_value_heads, layer->impl.head_size,
                    layer->impl.value_size, layer->impl.max_context, context->position);
            } else {
                qwen_attention_apply_online<<<layer->impl.heads, 256, 0, context->stream>>>(
                    static_cast<const float *>(device_query), layer->impl.key_cache,
                    layer->impl.key_cache_q8, layer->impl.value_cache, layer->impl.value_cache_q8,
                    static_cast<float *>(device_output), layer->impl.heads, layer->impl.key_value_heads,
                    layer->impl.head_size, layer->impl.value_size, layer->impl.max_context,
                    layer->impl.cache_key_type, layer->impl.cache_value_type, context->position);
            }
        }
        error = cudaGetLastError();
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen attention core device", error);
}

extern "C" int fortai_cuda_qwen35_attention_run_core_device_batch(
    fortai_cuda_qwen35_attention *layer, const void *device_query,
    size_t query_elements, const void *device_key, size_t key_elements,
    const void *device_value, size_t value_elements, int position, int batch,
    void *device_output, size_t output_elements) {
    if (!layer || !layer->impl.context || !device_query || !device_key ||
        !device_value || !device_output || position < 0 || batch <= 0 ||
        batch > INT32_MAX || position + batch > layer->impl.max_context)
        return FORTAI_CUDA_INVALID;
    auto *context = layer->impl.context;
    const size_t query_stride = static_cast<size_t>(layer->impl.heads) * 2 * layer->impl.head_size;
    const size_t key_stride = static_cast<size_t>(layer->impl.key_value_heads) * layer->impl.head_size;
    const size_t value_stride = static_cast<size_t>(layer->impl.key_value_heads) * layer->impl.value_size;
    const size_t attention_stride = static_cast<size_t>(layer->impl.heads) * layer->impl.value_size;
    if (query_elements < query_stride * batch || key_elements < key_stride * batch ||
        value_elements < value_stride * batch || output_elements < attention_stride * batch)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->device);
    cudaError_t error = cudaSuccess;
    qwen_attention_prepare_batch<<<dim3(static_cast<unsigned>(layer->impl.heads),
        static_cast<unsigned>(batch), 1), 256, 512 * sizeof(float), context->stream>>>(
        const_cast<float *>(static_cast<const float *>(device_query)),
        const_cast<float *>(static_cast<const float *>(device_key)),
        static_cast<const float *>(device_value), layer->impl.query_norm,
        layer->impl.key_norm, layer->impl.key_cache, layer->impl.value_cache,
        layer->impl.key_cache_q8, layer->impl.value_cache_q8,
        layer->impl.cache_key_type, layer->impl.cache_value_type, layer->impl.heads,
        layer->impl.key_value_heads, layer->impl.head_size, layer->impl.value_size,
        layer->impl.max_context, context->position, static_cast<int>(query_stride),
        static_cast<int>(key_stride), static_cast<int>(value_stride), batch,
        layer->impl.rope_dimension, layer->impl.rope_base, layer->impl.norm_epsilon);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
        if (layer->impl.cache_key_type == 0 && layer->impl.cache_value_type == 0 &&
            layer->impl.max_context <= 12000) {
            if (layer->impl.value_size <= 512) {
                const char *disable_mma = std::getenv("FORTAI_CUDA_ATTENTION_MMA");
                const bool use_mma = context->mma_available && layer->impl.head_size % 16 == 0 &&
                    layer->impl.head_size <= 512 && batch >= 16 &&
                    !(disable_mma && disable_mma[0] == '0');
                /* Qwen3.5 uses four query heads per KV head.  Process those
                 * four heads and four adjacent query tokens in one block so
                 * each K/V row is loaded once, matching the shape used by
                 * llama.cpp's grouped flash-attention kernel.  It is opt-in
                 * until it wins on the target's full-context profile; other
                 * shapes use the tuned tensor-core fallback.
                 */
                const char *enable_gqa = std::getenv("FORTAI_CUDA_ATTENTION_GQA");
                const bool use_gqa = context->mma_available &&
                    layer->impl.head_size == 256 && layer->impl.value_size == 256 &&
                    layer->impl.key_value_heads > 0 &&
                    layer->impl.heads == 4 * layer->impl.key_value_heads && batch >= 4 &&
                    enable_gqa && enable_gqa[0] == '1';
                if (use_gqa) {
                    constexpr int gqa_key_tile = 64;
                    constexpr size_t gqa_shared_bytes =
                        static_cast<size_t>(
                            4 * 16 * 256 + gqa_key_tile * 256 +
                            4 * 16 * gqa_key_tile + 8 * 16 * 16) * sizeof(__half) +
                        static_cast<size_t>(4 * 16 * gqa_key_tile) * sizeof(float);
                    error = cudaFuncSetAttribute(qwen_attention_apply_gqa_mma_f16_batch<4, 256, 256>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(gqa_shared_bytes));
                    if (error == cudaSuccess)
                        qwen_attention_apply_gqa_mma_f16_batch<4, 256, 256><<<dim3(
                            static_cast<unsigned>(layer->impl.key_value_heads),
                            static_cast<unsigned>((batch + 3) / 4), 1), 128,
                            gqa_shared_bytes, context->stream>>>(
                            static_cast<const float *>(device_query), layer->impl.key_cache,
                            layer->impl.value_cache, static_cast<float *>(device_output),
                            layer->impl.heads, layer->impl.key_value_heads, layer->impl.max_context,
                            position, static_cast<int>(query_stride),
                            static_cast<int>(attention_stride), batch, 2 * layer->impl.head_size);
                } else if (use_mma) {
                    const int query_shared_stride =
                        ((layer->impl.head_size + 15) / 16) * 16;
                    const size_t shared_bytes =
                        (static_cast<size_t>(16 * query_shared_stride + 64 * 16 +
                            16 * 64 + 8 * 16 * 16) * sizeof(__half)) +
                        static_cast<size_t>(16 * 64 + 16 * 3 +
                            8 * 16 * 16 + 16 * layer->impl.value_size) * sizeof(float);
                    error = cudaFuncSetAttribute(qwen_attention_apply_mma_f16_batch,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(shared_bytes));
                    if (error == cudaSuccess)
                        qwen_attention_apply_mma_f16_batch<<<dim3(
                            static_cast<unsigned>(layer->impl.heads),
                            static_cast<unsigned>((batch + 15) / 16), 1), 256, shared_bytes,
                            context->stream>>>(static_cast<const float *>(device_query),
                            layer->impl.key_cache, layer->impl.value_cache,
                            static_cast<float *>(device_output), layer->impl.heads,
                            layer->impl.key_value_heads, layer->impl.head_size,
                            layer->impl.value_size, layer->impl.max_context, position,
                            static_cast<int>(query_stride), static_cast<int>(attention_stride), batch,
                            2 * layer->impl.head_size);
                } else {
                    const size_t shared_bytes = static_cast<size_t>(64 +
                        8 * layer->impl.value_size) * sizeof(float);
                    qwen_attention_apply_flash_f16_batch<<<dim3(
                        static_cast<unsigned>(layer->impl.heads),
                        static_cast<unsigned>(batch), 1), 256, shared_bytes, context->stream>>>(
                        static_cast<const float *>(device_query), layer->impl.key_cache,
                        layer->impl.value_cache, static_cast<float *>(device_output),
                        layer->impl.heads, layer->impl.key_value_heads, layer->impl.head_size,
                        layer->impl.value_size, layer->impl.max_context, position,
                        static_cast<int>(query_stride), static_cast<int>(attention_stride), batch);
                }
            } else {
                qwen_attention_apply_f16_batch<<<dim3(static_cast<unsigned>(layer->impl.heads),
                    static_cast<unsigned>(batch), 1), 256,
                    static_cast<size_t>(8 + layer->impl.max_context) * sizeof(float), context->stream>>>(
                    static_cast<const float *>(device_query), layer->impl.key_cache,
                    layer->impl.value_cache, static_cast<float *>(device_output),
                    layer->impl.heads, layer->impl.key_value_heads, layer->impl.head_size,
                    layer->impl.value_size, layer->impl.max_context, position,
                    static_cast<int>(query_stride), static_cast<int>(attention_stride), batch);
            }
        } else {
            const bool use_q8_gqa = qwen_attention_q8_gqa_batch_enabled() &&
                layer->impl.cache_key_type == 1 && layer->impl.cache_value_type == 1 &&
                layer->impl.head_size == 256 && layer->impl.value_size == 256 &&
                layer->impl.key_value_heads > 0 &&
                (layer->impl.heads == 6 * layer->impl.key_value_heads ||
                 layer->impl.heads == 4 * layer->impl.key_value_heads);
            if (use_q8_gqa) {
                constexpr int q8_gqa_key_tile = 64;
                const int active_rows = position + batch;
                constexpr int fattn_kq_stride = 256;
                const int padded_rows = ((active_rows + fattn_kq_stride - 1) /
                    fattn_kq_stride) * fattn_kq_stride;
                bool use_q8_mma = qwen_attention_q8_mma_requested() &&
                    context->mma_available && batch >= 16;
                if (use_q8_mma) {
                    /* Dequantize the active prefix once, then reuse that view
                     * for all query tiles.  This is the same ownership model
                     * as llama.cpp's temporary K_f16/V_f16 pool and avoids
                     * decoding every Q/K/V tile repeatedly inside the MMA
                     * kernel. */
                    const size_t key_view_bytes = static_cast<size_t>(padded_rows) *
                        layer->impl.key_value_heads * layer->impl.head_size * sizeof(__half);
                    const size_t value_view_bytes = static_cast<size_t>(padded_rows) *
                        layer->impl.key_value_heads * layer->impl.value_size * sizeof(__half);
                    error = ensure_attention_f16_view(context, key_view_bytes, value_view_bytes);
                    if (error != cudaSuccess && std::getenv("FORTAI_CUDA_DEBUG"))
                        std::fprintf(stderr, "fortai-cuda: attention scratch: %s\n",
                            cudaGetErrorString(error));
                    if (error != cudaSuccess) {
                        /* Keep the quantized online kernel as a safe memory
                         * fallback when the active F16 view cannot be
                         * allocated at the requested context length. */
                        use_q8_mma = false;
                        error = cudaSuccess;
                    }
                    if (error == cudaSuccess && use_q8_mma) {
                        qwen_attention_dequantize_q8_view<256><<<
                            static_cast<unsigned>(active_rows * layer->impl.key_value_heads),
                            128, 0, context->stream>>>(layer->impl.key_cache_q8,
                            context->scratch_attention_key_f16, active_rows,
                            layer->impl.key_value_heads);
                        error = cudaGetLastError();
                        if (error != cudaSuccess && std::getenv("FORTAI_CUDA_DEBUG"))
                            std::fprintf(stderr, "fortai-cuda: attention key dequant: %s\n",
                                cudaGetErrorString(error));
                        if (error != cudaSuccess) {
                            use_q8_mma = false;
                            error = cudaSuccess;
                        }
                    }
                    if (error == cudaSuccess && use_q8_mma && padded_rows > active_rows) {
                        const size_t key_active_bytes = static_cast<size_t>(active_rows) *
                            layer->impl.key_value_heads * layer->impl.head_size * sizeof(__half);
                        error = cudaMemsetAsync(
                            reinterpret_cast<char *>(context->scratch_attention_key_f16) +
                                key_active_bytes,
                            0, key_view_bytes - key_active_bytes, context->stream);
                    }
                    if (error == cudaSuccess && use_q8_mma) {
                        qwen_attention_dequantize_q8_view<256><<<
                            static_cast<unsigned>(active_rows * layer->impl.key_value_heads),
                            128, 0, context->stream>>>(layer->impl.value_cache_q8,
                            context->scratch_attention_value_f16, active_rows,
                            layer->impl.key_value_heads);
                        error = cudaGetLastError();
                        if (error != cudaSuccess && std::getenv("FORTAI_CUDA_DEBUG"))
                            std::fprintf(stderr, "fortai-cuda: attention value dequant: %s\n",
                                cudaGetErrorString(error));
                        if (error != cudaSuccess) {
                            use_q8_mma = false;
                            error = cudaSuccess;
                        }
                    }
                    if (error == cudaSuccess && use_q8_mma && padded_rows > active_rows) {
                        const size_t value_active_bytes = static_cast<size_t>(active_rows) *
                            layer->impl.key_value_heads * layer->impl.value_size * sizeof(__half);
                        error = cudaMemsetAsync(
                            reinterpret_cast<char *>(context->scratch_attention_value_f16) +
                                value_active_bytes,
                            0, value_view_bytes - value_active_bytes, context->stream);
                    }
                }
                bool launched_fattn = false;
                if (use_q8_mma && layer->impl.heads == 6 * layer->impl.key_value_heads) {
                    const size_t mask_bytes = fortai_cuda_qwen38_fattn_mask_bytes(batch,
                        padded_rows);
                    const size_t meta_bytes = fortai_cuda_qwen38_fattn_meta_bytes(batch,
                        layer->impl.key_value_heads, padded_rows,
                        context->multiprocessor_count);
                    error = ensure_attention_fattn_scratch(context, mask_bytes, meta_bytes);
                    if (error == cudaSuccess) {
                        error = static_cast<cudaError_t>(fortai_cuda_qwen38_fattn_f16(
                            static_cast<const float *>(device_query),
                            context->scratch_attention_key_f16,
                            context->scratch_attention_value_f16,
                            static_cast<float *>(device_output),
                            context->scratch_attention_mask_f16,
                            context->scratch_attention_meta,
                            context->scratch_attention_meta_bytes,
                            layer->impl.heads, layer->impl.key_value_heads, position,
                            batch, padded_rows, static_cast<int>(query_stride),
                            static_cast<int>(attention_stride), 2 * layer->impl.head_size,
                            context->multiprocessor_count,
                            reinterpret_cast<void *>(context->stream)));
                    }
                    if (error == cudaSuccess) {
                        launched_fattn = true;
                    } else {
                        if (std::getenv("FORTAI_CUDA_DEBUG"))
                            std::fprintf(stderr, "fortai-cuda: licensed F16 fattn tile: %s\n",
                                cudaGetErrorString(error));
                        error = cudaSuccess;
                    }
                }
                if (launched_fattn) {
                    /* The licensed native tile includes stream-K fixup and
                     * the Qwen sigmoid gate. */
                } else if (use_q8_mma &&
                    layer->impl.heads == 6 * layer->impl.key_value_heads) {
                    /* The validated llama-shaped active-view MMA tile. */
                    constexpr int q8_mma_group = 6;
                    constexpr int q8_mma_query_tokens = 16;
                    constexpr int q8_mma_key_tile = 32;
                    constexpr size_t q8_mma_shared_bytes =
                        static_cast<size_t>(q8_mma_group * q8_mma_query_tokens * 256) * sizeof(__half) +
                        static_cast<size_t>(q8_mma_key_tile * 256) * sizeof(__half) +
                        static_cast<size_t>(q8_mma_group * q8_mma_query_tokens * q8_mma_key_tile) * sizeof(float) +
                        static_cast<size_t>(q8_mma_group * q8_mma_query_tokens * q8_mma_key_tile) * sizeof(__half) +
                        static_cast<size_t>(q8_mma_group * 16 * 16) * sizeof(__half);
                    error = cudaFuncSetAttribute(
                        qwen_attention_apply_gqa_mma_f16_batch<q8_mma_group, 256, 256,
                            q8_mma_query_tokens, q8_mma_key_tile>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(q8_mma_shared_bytes));
                    if (error == cudaSuccess)
                        qwen_attention_apply_gqa_mma_f16_batch<q8_mma_group, 256, 256,
                            q8_mma_query_tokens, q8_mma_key_tile><<<dim3(
                                static_cast<unsigned>(layer->impl.key_value_heads),
                                static_cast<unsigned>((batch + q8_mma_query_tokens - 1) /
                                    q8_mma_query_tokens), 1), q8_mma_group * 32,
                                q8_mma_shared_bytes, context->stream>>>(
                            static_cast<const float *>(device_query), context->scratch_attention_key_f16,
                            context->scratch_attention_value_f16, static_cast<float *>(device_output),
                            layer->impl.heads, layer->impl.key_value_heads, position + batch,
                            position, static_cast<int>(query_stride),
                            static_cast<int>(attention_stride), batch, 2 * layer->impl.head_size);
                } else if (use_q8_mma && layer->impl.heads == 4 * layer->impl.key_value_heads) {
                    constexpr int q8_mma_group = 4;
                    constexpr int q8_mma_query_tokens = 16;
                    constexpr int q8_mma_key_tile = 32;
                    constexpr size_t q8_mma_shared_bytes =
                        static_cast<size_t>(q8_mma_group * q8_mma_query_tokens * 256) * sizeof(__half) +
                        static_cast<size_t>(q8_mma_key_tile * 256) * sizeof(__half) +
                        static_cast<size_t>(q8_mma_group * q8_mma_query_tokens * q8_mma_key_tile) * sizeof(float) +
                        static_cast<size_t>(q8_mma_group * q8_mma_query_tokens * q8_mma_key_tile) * sizeof(__half) +
                        static_cast<size_t>(q8_mma_group * 16 * 16) * sizeof(__half);
                    error = cudaFuncSetAttribute(
                        qwen_attention_apply_gqa_mma_f16_batch<q8_mma_group, 256, 256,
                            q8_mma_query_tokens, q8_mma_key_tile>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(q8_mma_shared_bytes));
                    if (error == cudaSuccess)
                        qwen_attention_apply_gqa_mma_f16_batch<q8_mma_group, 256, 256,
                            q8_mma_query_tokens, q8_mma_key_tile><<<dim3(
                                static_cast<unsigned>(layer->impl.key_value_heads),
                                static_cast<unsigned>((batch + q8_mma_query_tokens - 1) /
                                    q8_mma_query_tokens), 1),
                            q8_mma_group * 32, q8_mma_shared_bytes, context->stream>>>(
                            static_cast<const float *>(device_query), context->scratch_attention_key_f16,
                            context->scratch_attention_value_f16, static_cast<float *>(device_output),
                            layer->impl.heads, layer->impl.key_value_heads, position + batch,
                            position, static_cast<int>(query_stride),
                            static_cast<int>(attention_stride), batch, 2 * layer->impl.head_size);
                } else if (layer->impl.heads == 6 * layer->impl.key_value_heads) {
                    constexpr int q8_group = 6;
                    constexpr int q8_query_tokens = 1;
                    const int q8_partitions = position >= 4096 ? 24 : 4;
                    constexpr size_t q8_gqa_shared_bytes =
                        static_cast<size_t>(q8_group * q8_query_tokens * 256) +
                        static_cast<size_t>(q8_group * q8_query_tokens * 8) * sizeof(float) +
                        static_cast<size_t>(q8_gqa_key_tile * (8 * q8_block_bytes) * 2) +
                        static_cast<size_t>(q8_group * q8_query_tokens * q8_gqa_key_tile) * sizeof(float);
                    const size_t partial_bytes = static_cast<size_t>(batch) *
                        layer->impl.heads * q8_partitions * (256 + 2) * sizeof(float);
                    error = ensure_attention_fattn_scratch(context, 0, partial_bytes);
                    if (error == cudaSuccess)
                        error = cudaFuncSetAttribute(qwen_attention_apply_gqa_q8_batch<q8_group,
                                256, 256, q8_query_tokens, q8_gqa_key_tile>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            static_cast<int>(q8_gqa_shared_bytes));
                    if (error == cudaSuccess)
                        qwen_attention_apply_gqa_q8_batch<q8_group, 256, 256,
                            q8_query_tokens, q8_gqa_key_tile><<<dim3(
                            static_cast<unsigned>(layer->impl.key_value_heads),
                            static_cast<unsigned>((batch + q8_query_tokens - 1) /
                                q8_query_tokens), q8_partitions), q8_group * 32,
                            q8_gqa_shared_bytes, context->stream>>>(
                            static_cast<const float *>(device_query), layer->impl.key_cache_q8,
                            layer->impl.value_cache_q8, static_cast<float *>(device_output),
                            layer->impl.heads, layer->impl.key_value_heads, layer->impl.max_context,
                            position, static_cast<int>(query_stride),
                            static_cast<int>(attention_stride), batch, context->position,
                            static_cast<float *>(context->scratch_attention_meta), q8_partitions);
                    if (error == cudaSuccess) {
                        error = cudaGetLastError();
                    }
                    if (error == cudaSuccess) {
                        qwen_attention_combine_q8_partitions<256><<<dim3(
                            static_cast<unsigned>(layer->impl.heads),
                            static_cast<unsigned>(batch), 1), 256, 0,
                            context->stream>>>(static_cast<const float *>(device_query),
                            static_cast<const float *>(context->scratch_attention_meta),
                            static_cast<float *>(device_output), layer->impl.heads, batch,
                            static_cast<int>(query_stride), static_cast<int>(attention_stride),
                            2 * layer->impl.head_size, context->position, q8_partitions);
                    }
                } else {
                    constexpr int q8_group = 4;
                    constexpr size_t q8_gqa_shared_bytes =
                        static_cast<size_t>(q8_group * 4 * 256) +
                        static_cast<size_t>(q8_group * 4 * 8) * sizeof(float) +
                        static_cast<size_t>(q8_gqa_key_tile * (8 * q8_block_bytes) * 2) +
                        static_cast<size_t>(q8_group * 4 * q8_gqa_key_tile) * sizeof(float);
                    error = cudaFuncSetAttribute(qwen_attention_apply_gqa_q8_batch<q8_group, 256, 256, 4, q8_gqa_key_tile>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(q8_gqa_shared_bytes));
                    if (error == cudaSuccess)
                        qwen_attention_apply_gqa_q8_batch<q8_group, 256, 256, 4, q8_gqa_key_tile><<<dim3(
                            static_cast<unsigned>(layer->impl.key_value_heads),
                            static_cast<unsigned>((batch + 3) / 4), 1), q8_group * 32,
                            q8_gqa_shared_bytes, context->stream>>>(
                            static_cast<const float *>(device_query), layer->impl.key_cache_q8,
                            layer->impl.value_cache_q8, static_cast<float *>(device_output),
                            layer->impl.heads, layer->impl.key_value_heads, layer->impl.max_context,
                            position, static_cast<int>(query_stride),
                            static_cast<int>(attention_stride), batch, context->position);
                }
            } else {
                /* Partitioning the key range buys parallelism only when the
                 * batch itself supplies few blocks, as in scalar decode.  A
                 * prefill ubatch already fills the GPU, and partitioning it
                 * would multiply the scratch buffer by the partition count for
                 * no gain.  The batch<=4 guard used to exclude prefill from the
                 * grouped q4 kernel entirely, sending every prompt chunk of the
                 * q4 draft cache through the unpartitioned scalar-oriented
                 * fallback: one block scanned the whole key range, which cost
                 * 233 ms per chunk at 16k tokens and dominated prefill. */
                const int online_partitions = batch <= 4 ? 8 : 1;
                const bool q4_cache = layer->impl.cache_key_type == 2 &&
                    layer->impl.cache_value_type == 2 &&
                    layer->impl.head_size == 256 && layer->impl.value_size == 256;
                /* Only a partitioned launch needs the shared scratch, and only
                 * scalar decode benefits from partitioning. */
                const bool partition_q4 = q4_cache && online_partitions > 1;
                const bool grouped_q4 = q4_cache &&
                    layer->impl.heads == 6 * layer->impl.key_value_heads;
                if (partition_q4) {
                    const size_t partial_bytes = static_cast<size_t>(batch) *
                        layer->impl.heads * online_partitions * (256 + 2) * sizeof(float);
                    error = ensure_attention_fattn_scratch(context, 0, partial_bytes);
                }
                if (error == cudaSuccess && grouped_q4) {
                    constexpr int q4_group = 6;
                    constexpr int q4_query_tokens = 1;
                    constexpr int q4_key_tile = 32;
                    constexpr size_t q4_shared_bytes =
                        static_cast<size_t>(q4_key_tile * (8 * q4_block_bytes) * 2) +
                        static_cast<size_t>(q4_group * q4_query_tokens * q4_key_tile) * sizeof(float);
                    qwen_attention_apply_gqa_q4_batch<q4_group, q4_query_tokens,
                        q4_key_tile><<<dim3(
                        static_cast<unsigned>(layer->impl.key_value_heads),
                        static_cast<unsigned>((batch + q4_query_tokens - 1) /
                            q4_query_tokens), online_partitions), q4_group * 32,
                        q4_shared_bytes, context->stream>>>(
                        static_cast<const float *>(device_query), layer->impl.key_cache_q8,
                        layer->impl.value_cache_q8, static_cast<float *>(device_output),
                        layer->impl.heads, layer->impl.key_value_heads,
                        layer->impl.max_context, context->position,
                        static_cast<int>(query_stride), static_cast<int>(attention_stride),
                        batch, partition_q4 ?
                            static_cast<float *>(context->scratch_attention_meta) : nullptr,
                        online_partitions);
                    error = cudaGetLastError();
                } else if (error == cudaSuccess) {
                    qwen_attention_apply_online_batch<<<dim3(
                        static_cast<unsigned>(layer->impl.heads),
                        static_cast<unsigned>(batch),
                        static_cast<unsigned>(partition_q4 ? online_partitions : 1)),
                        256, 0, context->stream>>>(
                        static_cast<const float *>(device_query), layer->impl.key_cache,
                        layer->impl.key_cache_q8, layer->impl.value_cache,
                        layer->impl.value_cache_q8, static_cast<float *>(device_output),
                        layer->impl.heads, layer->impl.key_value_heads,
                        layer->impl.head_size, layer->impl.value_size,
                        layer->impl.max_context, layer->impl.cache_key_type,
                        layer->impl.cache_value_type, context->position,
                        static_cast<int>(query_stride),
                        static_cast<int>(attention_stride), batch,
                        partition_q4 ?
                            static_cast<float *>(context->scratch_attention_meta) : nullptr,
                        partition_q4 ? online_partitions : 1);
                    error = cudaGetLastError();
                }
                if (error == cudaSuccess && partition_q4) {
                    qwen_attention_combine_q8_partitions<256><<<dim3(
                        static_cast<unsigned>(layer->impl.heads),
                        static_cast<unsigned>(batch), 1), 256, 0,
                        context->stream>>>(static_cast<const float *>(device_query),
                        static_cast<const float *>(context->scratch_attention_meta),
                        static_cast<float *>(device_output), layer->impl.heads, batch,
                        static_cast<int>(query_stride), static_cast<int>(attention_stride),
                        2 * layer->impl.head_size, context->position, online_partitions);
                }
            }
        }
        if (error == cudaSuccess) error = cudaGetLastError();
    }
    if (error != cudaSuccess && std::getenv("FORTAI_CUDA_DEBUG"))
        std::fprintf(stderr, "fortai-cuda: batched attention launch: %s\n",
            cudaGetErrorString(error));
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen attention batched core", error);
}

extern "C" const char *fortai_cuda_q8_last_error(
    const fortai_cuda_q8_context *context) {
    return context ? context->impl.error : "invalid CUDA context";
}
