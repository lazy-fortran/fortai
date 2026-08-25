#include "fortai_cuda_backend.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <new>
#include <vector>

namespace {

constexpr int q8_block_width = 32;
constexpr int q8_block_bytes = 34;
constexpr int q8_activation_block_bytes = 36;
constexpr int q8_activation_data_offset = 4;

struct fortai_cuda_q8_context_impl {
    int device = 0;
    cudaStream_t stream = nullptr;
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
    int *position = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
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

__global__ void q8_gemv_one_warp(const int8_t *__restrict__ weights,
    const int8_t *__restrict__ activation, int rows, int blocks,
    float *__restrict__ output) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= 32) return;
    const int8_t *row_weights = weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    float accumulator = 0.0f;
    for (int block = lane; block < blocks; block += 32) {
        const int8_t *weight_block = row_weights + block * q8_block_bytes;
        const int8_t *activation_block = activation + block * q8_activation_block_bytes;
        const float scale = block_scale(weight_block) * block_scale(activation_block);
        int dot = 0;
#pragma unroll
        for (int i = 0; i < q8_block_width; i += 4)
            dot = __dp4a(load_i32(weight_block + 2 + i),
                load_i32(activation_block + q8_activation_data_offset + i), dot);
        accumulator = fmaf(scale, static_cast<float>(dot), accumulator);
    }
    for (int offset = 16; offset; offset >>= 1)
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    if (lane == 0) output[row] = accumulator;
}

__global__ void q8_gemv_four_warp(const int8_t *__restrict__ weights,
    const int8_t *__restrict__ activation, int rows, int blocks,
    float *__restrict__ output) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (row >= rows) return;
    const int8_t *row_weights = weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    float accumulator = 0.0f;
    /* Four warps cooperate on one row.  Each lane owns complete 32-value
     * Q8 blocks; this keeps the shuffle reduction within one warp and avoids
     * mixing the old four-lane subgroups during the final sum. */
    for (int block = warp * 32 + lane; block < blocks; block += 128) {
        const int8_t *weight_block = row_weights + block * q8_block_bytes;
        const int8_t *activation_block = activation + block * q8_activation_block_bytes;
        const float scale = block_scale(weight_block) * block_scale(activation_block);
        int dot = 0;
#pragma unroll
        for (int i = 0; i < q8_block_width; i += 4) {
            const int offset = 2 + i;
            dot = __dp4a(load_i32(weight_block + offset),
                load_i32(activation_block + q8_activation_data_offset + i), dot);
        }
        accumulator = fmaf(scale, static_cast<float>(dot), accumulator);
    }
    for (int offset = 16; offset; offset >>= 1)
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    __shared__ float partial[4];
    if (lane == 0) partial[warp] = accumulator;
    __syncthreads();
    if (warp == 0) {
        accumulator = lane < 4 ? partial[lane] : 0.0f;
        for (int offset = 2; offset; offset >>= 1)
            accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
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
    if (weights->blocks >= 128)
        q8_gemv_four_warp<<<weights->rows, 128, 0, stream>>>(weights->device_data,
            activation, weights->rows, weights->blocks, output);
    else
        q8_gemv_one_warp<<<weights->rows, 32, 0, stream>>>(weights->device_data,
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

static cudaError_t ensure_aux_scratch(fortai_cuda_q8_context_impl *context, size_t bytes) {
    if (bytes <= context->scratch_aux_bytes) return cudaSuccess;
    cudaError_t error = cudaFree(context->scratch_aux);
    if (error != cudaSuccess) return error;
    context->scratch_aux = nullptr;
    error = cudaMalloc(reinterpret_cast<void **>(&context->scratch_aux), bytes);
    if (error == cudaSuccess) context->scratch_aux_bytes = bytes;
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

__global__ void qwen_silu_product(float *gate, const float *up, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    gate[index] = qwen_silu(gate[index]) * up[index];
}

__global__ void q8_gemv_silu_product_four_warp(const int8_t *gate_weights,
    const int8_t *up_weights, const int8_t *activation, float *output,
    int rows, int blocks) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (row >= rows) return;
    const int8_t *gate_row = gate_weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    const int8_t *up_row = up_weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    float gate_accumulator = 0.0f;
    float up_accumulator = 0.0f;
    for (int block = warp * 32 + lane; block < blocks; block += 128) {
        const int8_t *gate_block = gate_row + block * q8_block_bytes;
        const int8_t *up_block = up_row + block * q8_block_bytes;
        const int8_t *activation_block = activation + block * q8_activation_block_bytes;
        const float activation_scale = block_scale(activation_block);
        const float gate_scale = block_scale(gate_block) * activation_scale;
        const float up_scale = block_scale(up_block) * activation_scale;
        int gate_dot = 0;
        int up_dot = 0;
#pragma unroll
        for (int i = 0; i < q8_block_width; i += 4) {
            const int offset = 2 + i;
            const int activation_values = load_i32(activation_block + q8_activation_data_offset + i);
            gate_dot = __dp4a(load_i32(gate_block + offset), activation_values, gate_dot);
            up_dot = __dp4a(load_i32(up_block + offset), activation_values, up_dot);
        }
        gate_accumulator = fmaf(gate_scale, static_cast<float>(gate_dot), gate_accumulator);
        up_accumulator = fmaf(up_scale, static_cast<float>(up_dot), up_accumulator);
    }
    for (int offset = 16; offset; offset >>= 1) {
        gate_accumulator += __shfl_down_sync(0xffffffffu, gate_accumulator, offset);
        up_accumulator += __shfl_down_sync(0xffffffffu, up_accumulator, offset);
    }
    __shared__ float gate_partial[4];
    __shared__ float up_partial[4];
    if (lane == 0) {
        gate_partial[warp] = gate_accumulator;
        up_partial[warp] = up_accumulator;
    }
    __syncthreads();
    if (warp == 0) {
        gate_accumulator = lane < 4 ? gate_partial[lane] : 0.0f;
        up_accumulator = lane < 4 ? up_partial[lane] : 0.0f;
        for (int offset = 2; offset; offset >>= 1) {
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
    for (int block = lane; block < blocks; block += 32) {
        const int8_t *gate_block = gate_row + block * q8_block_bytes;
        const int8_t *up_block = up_row + block * q8_block_bytes;
        const int8_t *activation_block = activation + block * q8_activation_block_bytes;
        const float gate_scale = block_scale(gate_block) * block_scale(activation_block);
        const float up_scale = block_scale(up_block) * block_scale(activation_block);
        int gate_dot = 0;
        int up_dot = 0;
#pragma unroll
        for (int i = 0; i < q8_block_width; i += 4) {
            const int activation_values = load_i32(activation_block + q8_activation_data_offset + i);
            gate_dot = __dp4a(load_i32(gate_block + 2 + i), activation_values, gate_dot);
            up_dot = __dp4a(load_i32(up_block + 2 + i), activation_values, up_dot);
        }
        gate_accumulator = fmaf(gate_scale, static_cast<float>(gate_dot), gate_accumulator);
        up_accumulator = fmaf(up_scale, static_cast<float>(up_dot), up_accumulator);
    }
    for (int offset = 16; offset; offset >>= 1) {
        gate_accumulator += __shfl_down_sync(0xffffffffu, gate_accumulator, offset);
        up_accumulator += __shfl_down_sync(0xffffffffu, up_accumulator, offset);
    }
    if (lane == 0) output[row] = qwen_silu(gate_accumulator) * up_accumulator;
}

static void launch_q8_silu_product(fortai_cuda_q8_weights_impl *gate,
    fortai_cuda_q8_weights_impl *up, const int8_t *activation, float *output,
    float *auxiliary, cudaStream_t stream) {
    if (gate->blocks <= 96) {
        q8_gemv_silu_product_one_warp<<<gate->rows, 32, 0, stream>>>(gate->device_data,
            up->device_data, activation, output, gate->rows, gate->blocks);
    } else if (gate->blocks < 128) {
        q8_gemv_silu_product_four_warp<<<gate->rows, 128, 0, stream>>>(gate->device_data,
            up->device_data, activation, output, gate->rows, gate->blocks);
    } else {
        launch_q8(gate, activation, output, stream);
        launch_q8(up, activation, auxiliary, stream);
        qwen_silu_product<<<(gate->rows + 255) / 256, 256, 0, stream>>>(
            output, auxiliary, gate->rows);
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
    if (slice >= slices) return;
    extern __shared__ float partial[];
    const int lane = static_cast<int>(threadIdx.x);
    float sum = 0.0f;
    for (int i = lane; i < length; i += blockDim.x) {
        const float value = values[slice * length + i];
        sum += value * value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float inverse = 1.0f / fmaxf(sqrtf(partial[0]), epsilon);
    for (int i = lane; i < length; i += blockDim.x)
        values[slice * length + i] *= inverse;
}

__global__ void qwen_recurrent_gdn_rows(const float *qkv, const float *gate,
    const float *alpha, const float *beta, const float *ssm_a, const float *ssm_dt,
    float *state, float *output, int state_size, int key_heads, int value_heads,
    int head_size) {
    const int row_index = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int head = row_index / head_size;
    const int row = row_index % head_size;
    if (head >= value_heads || lane >= 32) return;
    const int key_head = head % key_heads;
    const int q_offset = key_head * head_size;
    const int k_offset = state_size * key_heads + key_head * head_size;
    const int v_offset = 2 * state_size * key_heads + head * head_size;
    const float decay = expf(ssm_a[head] * qwen_softplus(alpha[head] + ssm_dt[head]));
    const float beta_value = qwen_sigmoid(beta[head]);
    float *head_state = state + head * head_size * head_size;
    float key_dot = 0.0f;
    for (int column = lane; column < head_size; column += 32) {
        const int index = row * head_size + column;
        const float decayed = head_state[index] * decay;
        head_state[index] = decayed;
        key_dot += decayed * qkv[k_offset + column];
    }
    for (int offset = 16; offset; offset >>= 1)
        key_dot += __shfl_down_sync(0xffffffffu, key_dot, offset);
    key_dot = __shfl_sync(0xffffffffu, key_dot, 0);
    __shared__ float row_delta;
    if (lane == 0) row_delta = (qkv[v_offset + row] - key_dot) * beta_value;
    __syncthreads();
    float query_dot = 0.0f;
    for (int column = lane; column < head_size; column += 32) {
        const int index = row * head_size + column;
        const float updated = head_state[index] + row_delta * qkv[k_offset + column];
        head_state[index] = updated;
        query_dot += updated * qkv[q_offset + column];
    }
    for (int offset = 16; offset; offset >>= 1)
        query_dot += __shfl_down_sync(0xffffffffu, query_dot, offset);
    if (lane == 0) output[row_index] = query_dot / sqrtf(static_cast<float>(head_size));
}

__global__ void qwen_recurrent_gdn_normalize(float *output, const float *gate,
    const float *ssm_norm, int value_heads, int head_size, float norm_epsilon) {
    const int head = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (head >= value_heads || tid >= head_size) return;
    extern __shared__ float reduction[];
    const int offset = head * head_size + tid;
    reduction[tid] = output[offset] * output[offset];
    __syncthreads();
    for (int stride = head_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    const float inverse = 1.0f / sqrtf(reduction[0] / static_cast<float>(head_size) + norm_epsilon);
    output[offset] = output[offset] * inverse * ssm_norm[tid] * qwen_silu(gate[offset]);
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

__global__ void qwen_add_float(const float *left, const float *right, float *output,
    int elements) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) output[index] = left[index] + right[index];
}

__global__ void qwen_rms_norm_float(const float *input, const float *weights,
    float *output, int elements, float epsilon) {
    extern __shared__ float norm_partial[];
    const int lane = static_cast<int>(threadIdx.x);
    float sum = 0.0f;
    for (int index = lane; index < elements; index += blockDim.x) {
        const float value = input[index];
        sum += value * value;
    }
    norm_partial[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) norm_partial[lane] += norm_partial[lane + stride];
        __syncthreads();
    }
    const float inverse = 1.0f / sqrtf(norm_partial[0] / static_cast<float>(elements) + epsilon);
    for (int index = lane; index < elements; index += blockDim.x)
        output[index] = input[index] * inverse * weights[index];
}

struct qwen_argmax_pair {
    float value;
    int index;
};

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
    __half *value_cache, int heads, int key_value_heads, int head_size,
    int value_size, int max_context, const int *position_ptr, int rope_dimension,
    float rope_base, float epsilon) {
    const int head = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (head >= heads) return;
    const int position = *position_ptr;
    extern __shared__ double attention_partial[];
    const int query_offset = head * 2 * head_size;
    double query_sum = 0.0;
    for (int i = tid; i < head_size; i += blockDim.x) {
        const float x = query[query_offset + i];
        query_sum += static_cast<double>(x) * static_cast<double>(x);
    }
    attention_partial[tid] = query_sum;
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const bool writes_kv = (head % group) == 0;
    double key_sum = 0.0;
    const int key_offset = kv_head * head_size;
    if (writes_kv) {
        for (int i = tid; i < head_size; i += blockDim.x) {
            const float x = key[key_offset + i];
            key_sum += static_cast<double>(x) * static_cast<double>(x);
        }
    }
    attention_partial[blockDim.x + tid] = key_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            attention_partial[tid] += attention_partial[tid + stride];
            attention_partial[blockDim.x + tid] += attention_partial[blockDim.x + tid + stride];
        }
        __syncthreads();
    }
    const float query_inverse = 1.0f / sqrtf(
        static_cast<float>(attention_partial[0] / static_cast<double>(head_size)) + epsilon);
    for (int i = tid; i < head_size; i += blockDim.x)
        query[query_offset + i] *= query_inverse * query_norm[i];
    __syncthreads();
    if (rope_dimension > 0) {
        for (int i = tid; i < rope_dimension / 2; i += blockDim.x)
            qwen_rope_pair(query + query_offset, rope_dimension, i, position, rope_base);
    }
    if (writes_kv) {
        const float key_inverse = 1.0f / sqrtf(
            static_cast<float>(attention_partial[blockDim.x] / static_cast<double>(head_size)) + epsilon);
        for (int i = tid; i < head_size; i += blockDim.x)
            key[key_offset + i] *= key_inverse * key_norm[i];
        __syncthreads();
        if (rope_dimension > 0) {
            for (int i = tid; i < rope_dimension / 2; i += blockDim.x)
                qwen_rope_pair(key + key_offset, rope_dimension, i, position, rope_base);
        }
        for (int i = tid; i < head_size; i += blockDim.x)
            key_cache[position * key_value_heads * head_size + key_offset + i] =
                __float2half_rn(key[key_offset + i]);
        for (int i = tid; i < value_size; i += blockDim.x)
            value_cache[position * key_value_heads * value_size + kv_head * value_size + i] =
                __float2half_rn(value[kv_head * value_size + i]);
    }
}

__global__ void qwen_attention_apply(const float *query, const __half *key_cache,
    const __half *value_cache, float *attention, int heads, int key_value_heads,
    int head_size, int value_size, int max_context, const int *position_ptr) {
    const int head = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (head >= heads) return;
    const int position = *position_ptr;
    extern __shared__ float shared[];
    float *warps = shared;
    float *scores = shared + 8;
    const int group = heads / key_value_heads;
    const int kv_head = head / group;
    const int query_offset = head * 2 * head_size;
    for (int previous = 0; previous <= position; ++previous) {
        float dot = 0.0f;
        const int key_offset = (previous * key_value_heads + kv_head) * head_size;
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
        float maximum = -3.402823466e+38f;
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
            const int value_offset = (previous * key_value_heads + kv_head) * value_size;
            total += scores[previous] * __half2float(value_cache[value_offset + i]);
        }
        attention[head * value_size + i] = total * qwen_sigmoid(gate[i]);
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
};

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
    cudaFree(layer->query);
    cudaFree(layer->key);
    cudaFree(layer->value);
    cudaFree(layer->attention);
    layer->query_norm = nullptr;
    layer->key_norm = nullptr;
    layer->key_cache = nullptr;
    layer->value_cache = nullptr;
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
    if (cudaSetDevice(device) != cudaSuccess) return FORTAI_CUDA_RUNTIME_ERROR;
    const cudaError_t error = cudaMemGetInfo(free_bytes, total_bytes);
    return error == cudaSuccess ? FORTAI_CUDA_OK : FORTAI_CUDA_RUNTIME_ERROR;
}

extern "C" int fortai_cuda_q8_context_destroy(fortai_cuda_q8_context *context) {
    if (!context) return FORTAI_CUDA_OK;
    cudaSetDevice(context->impl.device);
    cudaGraphExecDestroy(context->impl.graph_exec);
    cudaGraphDestroy(context->impl.graph);
    cudaFree(context->impl.position);
    cudaFree(context->impl.scratch_activation);
    cudaFree(context->impl.scratch_raw);
    cudaFree(context->impl.scratch_output);
    cudaFree(context->impl.scratch_aux);
    cudaEventDestroy(context->impl.start);
    cudaEventDestroy(context->impl.stop);
    cudaStreamDestroy(context->impl.stream);
    delete context;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_q8_context_set_position(fortai_cuda_q8_context *context, int position) {
    if (!context || !context->impl.position || position < 0) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaMemcpyAsync(context->impl.position, &position, sizeof(position),
        cudaMemcpyHostToDevice, context->impl.stream);
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

extern "C" int fortai_cuda_q8_context_capture_begin(fortai_cuda_q8_context *context) {
    if (!context || context->impl.graph || context->impl.graph_exec) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaStreamBeginCapture(context->impl.stream,
        cudaStreamCaptureModeGlobal);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "begin CUDA graph capture", error);
}

extern "C" int fortai_cuda_q8_context_capture_end(fortai_cuda_q8_context *context) {
    if (!context) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    cudaError_t error = cudaStreamEndCapture(context->impl.stream, &context->impl.graph);
    if (error == cudaSuccess)
        error = cudaGraphInstantiate(&context->impl.graph_exec, context->impl.graph,
            nullptr, nullptr, 0);
    if (error != cudaSuccess) {
        cudaGraphExecDestroy(context->impl.graph_exec);
        cudaGraphDestroy(context->impl.graph);
        context->impl.graph_exec = nullptr;
        context->impl.graph = nullptr;
        return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "end CUDA graph capture", error);
    }
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_q8_context_graph_launch(fortai_cuda_q8_context *context) {
    if (!context || !context->impl.graph_exec) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    const cudaError_t error = cudaGraphLaunch(context->impl.graph_exec, context->impl.stream);
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
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->impl.stream);
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
    const cudaError_t error = cudaMemcpyAsync(host_data, device_buffer, bytes,
        cudaMemcpyDeviceToHost, context->impl.stream);
    if (error == cudaSuccess) {
        const cudaError_t sync = cudaStreamSynchronize(context->impl.stream);
        if (sync != cudaSuccess)
            return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device download", sync);
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
    const size_t activation_bytes = static_cast<size_t>(blocks) * q8_block_bytes;
    const size_t output_bytes = static_cast<size_t>(weights->impl.rows) * sizeof(float);
    cudaSetDevice(context->impl.device);
    cudaError_t error = ensure_host_matvec_scratch(&context->impl, activation_bytes, output_bytes);
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

extern "C" int fortai_cuda_qwen35_rms_norm_device(fortai_cuda_q8_context *context,
    const void *device_input, const void *device_weights, void *device_output,
    size_t elements, float epsilon) {
    if (!context || !device_input || !device_weights || !device_output || elements == 0 ||
        elements > static_cast<size_t>(INT32_MAX) || epsilon <= 0.0f)
        return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->impl.device);
    qwen_rms_norm_float<<<1, 256, 256 * sizeof(float), context->impl.stream>>>(
        static_cast<const float *>(device_input), static_cast<const float *>(device_weights),
        static_cast<float *>(device_output), static_cast<int>(elements), epsilon);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device RMS norm", error);
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
    std::vector<qwen_argmax_pair> partials(static_cast<size_t>(blocks));
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(partials.data(), context->impl.scratch_aux, bytes,
            cudaMemcpyDeviceToHost, context->impl.stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->impl.stream);
    if (error != cudaSuccess)
        return fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "device argmax", error);
    qwen_argmax_pair best = {-FLT_MAX, INT_MAX};
    for (const qwen_argmax_pair candidate : partials) {
        if (candidate.value > best.value ||
            (candidate.value == best.value && candidate.index < best.index)) best = candidate;
    }
    *host_index = best.index;
    return FORTAI_CUDA_OK;
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
            context->impl.scratch_output, context->impl.scratch_aux, context->impl.stream);
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
            context->impl.scratch_output, context->impl.scratch_aux, context->impl.stream);
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
        if (error == cudaSuccess) error = cudaStreamSynchronize(context->impl.stream);
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
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.gdn_output),
        static_cast<size_t>(inner_size) * sizeof(float));
    if (error == cudaSuccess) error = cudaMemsetAsync(created->impl.conv_state, 0,
        static_cast<size_t>(conv_kernel - 1) * conv_size * sizeof(float), context->impl.stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(created->impl.gdn_state, 0,
        static_cast<size_t>(value_heads) * head_size * head_size * sizeof(float), context->impl.stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->impl.stream);
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
        qwen_recurrent_l2_normalize<<<slices, 128, 128 * sizeof(float), context->stream>>>(
            layer->impl.qkv_output, slices, layer->impl.head_size, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_recurrent_gdn_rows<<<layer->impl.value_heads * layer->impl.head_size,
            32, sizeof(float), context->stream>>>(
            layer->impl.qkv_output, layer->impl.gate_output, layer->impl.alpha_output,
            layer->impl.beta_output, layer->impl.ssm_a, layer->impl.ssm_dt,
            layer->impl.gdn_state, layer->impl.gdn_output,
            layer->impl.state_size, layer->impl.key_heads, layer->impl.value_heads,
            layer->impl.head_size);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_recurrent_gdn_normalize<<<layer->impl.value_heads, layer->impl.head_size,
            static_cast<size_t>(layer->impl.head_size) * sizeof(float), context->stream>>>(
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
        qwen_recurrent_l2_normalize<<<slices, 128, 128 * sizeof(float), context->stream>>>(
            layer->impl.qkv_output, slices, layer->impl.head_size, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_recurrent_gdn_rows<<<layer->impl.value_heads * layer->impl.head_size,
            32, sizeof(float), context->stream>>>(
            layer->impl.qkv_output, layer->impl.gate_output, layer->impl.alpha_output,
            layer->impl.beta_output, layer->impl.ssm_a, layer->impl.ssm_dt,
            layer->impl.gdn_state, layer->impl.gdn_output,
            layer->impl.state_size, layer->impl.key_heads, layer->impl.value_heads,
            layer->impl.head_size);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_recurrent_gdn_normalize<<<layer->impl.value_heads, layer->impl.head_size,
            static_cast<size_t>(layer->impl.head_size) * sizeof(float), context->stream>>>(
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
        qwen_recurrent_l2_normalize<<<slices, 128, 128 * sizeof(float), context->stream>>>(
            static_cast<float *>(device_qkv), slices, layer->impl.head_size, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_recurrent_gdn_rows<<<layer->impl.value_heads * layer->impl.head_size,
            32, sizeof(float), context->stream>>>(static_cast<const float *>(device_qkv),
            static_cast<const float *>(device_gate), static_cast<const float *>(device_alpha),
            static_cast<const float *>(device_beta), layer->impl.ssm_a, layer->impl.ssm_dt,
            layer->impl.gdn_state, layer->impl.gdn_output, layer->impl.state_size,
            layer->impl.key_heads, layer->impl.value_heads, layer->impl.head_size);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_recurrent_gdn_normalize<<<layer->impl.value_heads, layer->impl.head_size,
            static_cast<size_t>(layer->impl.head_size) * sizeof(float), context->stream>>>(
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

extern "C" int fortai_cuda_qwen35_attention_create(
    fortai_cuda_q8_context *context, const fortai_cuda_q8_weights *query_weights,
    const fortai_cuda_q8_weights *key_weights, const fortai_cuda_q8_weights *value_weights,
    const fortai_cuda_q8_weights *output_weights, const void *query_norm,
    size_t query_norm_bytes, const void *key_norm, size_t key_norm_bytes,
    int heads, int key_value_heads, int head_size, int value_size, int max_context,
    int rope_dimension, float rope_base, float norm_epsilon,
    fortai_cuda_qwen35_attention **layer) {
    if (!context || !layer || !query_weights || !key_weights || !value_weights ||
        !output_weights || !query_norm || !key_norm || heads <= 0 || key_value_heads <= 0 ||
        head_size <= 0 || value_size <= 0 || max_context <= 0 ||
        heads % key_value_heads != 0 || rope_dimension < 0 ||
        (rope_dimension > 0 && (rope_dimension > head_size || rope_dimension % 2 != 0)) ||
        rope_base <= 0.0f || norm_epsilon <= 0.0f)
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
    cudaSetDevice(context->impl.device);
    cudaError_t error = allocate_and_copy_float_attention(&created->impl,
        &created->impl.query_norm, query_norm, static_cast<size_t>(head_size) * sizeof(float));
    if (error == cudaSuccess) error = allocate_and_copy_float_attention(&created->impl,
        &created->impl.key_norm, key_norm, static_cast<size_t>(head_size) * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.key_cache),
        static_cast<size_t>(max_context) * key_value_heads * head_size * sizeof(__half));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.value_cache),
        static_cast<size_t>(max_context) * key_value_heads * value_size * sizeof(__half));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.query),
        static_cast<size_t>(heads) * 2 * head_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.key),
        static_cast<size_t>(key_value_heads) * head_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.value),
        static_cast<size_t>(key_value_heads) * value_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.attention),
        static_cast<size_t>(heads) * value_size * sizeof(float));
    if (error == cudaSuccess) error = cudaMemsetAsync(created->impl.key_cache, 0,
        static_cast<size_t>(max_context) * key_value_heads * head_size * sizeof(__half),
        context->impl.stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(created->impl.value_cache, 0,
        static_cast<size_t>(max_context) * key_value_heads * value_size * sizeof(__half),
        context->impl.stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->impl.stream);
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
    float rope_base, float norm_epsilon, fortai_cuda_qwen35_attention **layer) {
    if (!context || !layer || !query_norm || !key_norm || heads <= 0 ||
        key_value_heads <= 0 || head_size <= 0 || value_size <= 0 || max_context <= 0 ||
        heads % key_value_heads != 0 || rope_dimension < 0 ||
        (rope_dimension > 0 && (rope_dimension > head_size || rope_dimension % 2 != 0)) ||
        rope_base <= 0.0f || norm_epsilon <= 0.0f ||
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
    cudaSetDevice(context->impl.device);
    cudaError_t error = allocate_and_copy_float_attention(&created->impl,
        &created->impl.query_norm, query_norm, static_cast<size_t>(head_size) * sizeof(float));
    if (error == cudaSuccess) error = allocate_and_copy_float_attention(&created->impl,
        &created->impl.key_norm, key_norm, static_cast<size_t>(head_size) * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.key_cache),
        static_cast<size_t>(max_context) * key_value_heads * head_size * sizeof(__half));
    if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void **>(&created->impl.value_cache),
        static_cast<size_t>(max_context) * key_value_heads * value_size * sizeof(__half));
    if (error == cudaSuccess) error = cudaMemsetAsync(created->impl.key_cache, 0,
        static_cast<size_t>(max_context) * key_value_heads * head_size * sizeof(__half), context->impl.stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(created->impl.value_cache, 0,
        static_cast<size_t>(max_context) * key_value_heads * value_size * sizeof(__half), context->impl.stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->impl.stream);
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
    cudaError_t error = cudaMemsetAsync(layer->impl.key_cache, 0,
        static_cast<size_t>(layer->impl.max_context) * layer->impl.key_value_heads *
            layer->impl.head_size * sizeof(__half), context->stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(layer->impl.value_cache, 0,
        static_cast<size_t>(layer->impl.max_context) * layer->impl.key_value_heads *
            layer->impl.value_size * sizeof(__half), context->stream);
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
        qwen_attention_prepare<<<layer->impl.heads, 256, 512 * sizeof(double), context->stream>>>(
            layer->impl.query, layer->impl.key, layer->impl.value, layer->impl.query_norm,
            layer->impl.key_norm, layer->impl.key_cache, layer->impl.value_cache,
            layer->impl.heads, layer->impl.key_value_heads, layer->impl.head_size,
            layer->impl.value_size, layer->impl.max_context, context->position,
            layer->impl.rope_dimension, layer->impl.rope_base, layer->impl.norm_epsilon);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        qwen_attention_apply<<<layer->impl.heads, 256,
            static_cast<size_t>(8 + layer->impl.max_context) * sizeof(float), context->stream>>>(
            layer->impl.query, layer->impl.key_cache, layer->impl.value_cache,
            layer->impl.attention, layer->impl.heads, layer->impl.key_value_heads,
            layer->impl.head_size, layer->impl.value_size, layer->impl.max_context, context->position);
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
    qwen_attention_prepare<<<layer->impl.heads, 256, 512 * sizeof(double), context->stream>>>(
        const_cast<float *>(static_cast<const float *>(device_query)),
        const_cast<float *>(static_cast<const float *>(device_key)),
        static_cast<const float *>(device_value), layer->impl.query_norm, layer->impl.key_norm,
        layer->impl.key_cache, layer->impl.value_cache, layer->impl.heads,
        layer->impl.key_value_heads, layer->impl.head_size, layer->impl.value_size,
        layer->impl.max_context, context->position, layer->impl.rope_dimension,
        layer->impl.rope_base, layer->impl.norm_epsilon);
    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        qwen_attention_apply<<<layer->impl.heads, 256,
            static_cast<size_t>(8 + layer->impl.max_context) * sizeof(float), context->stream>>>(
            static_cast<const float *>(device_query), layer->impl.key_cache,
            layer->impl.value_cache, static_cast<float *>(device_output), layer->impl.heads,
            layer->impl.key_value_heads, layer->impl.head_size, layer->impl.value_size,
            layer->impl.max_context, context->position);
        error = cudaGetLastError();
    }
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(context, FORTAI_CUDA_RUNTIME_ERROR, "Qwen attention core device", error);
}

extern "C" const char *fortai_cuda_q8_last_error(
    const fortai_cuda_q8_context *context) {
    return context ? context->impl.error : "invalid CUDA context";
}
