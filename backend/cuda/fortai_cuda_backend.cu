#include "fortai_cuda_backend.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <new>

namespace {

constexpr int q8_block_width = 32;
constexpr int q8_block_bytes = 34;

struct fortai_cuda_q8_context_impl {
    int device = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    char error[256] = {};
};

struct fortai_cuda_q8_weights_impl {
    fortai_cuda_q8_context_impl *context = nullptr;
    int rows = 0;
    int width = 0;
    int blocks = 0;
    int8_t *device_data = nullptr;
};

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
        const int8_t *activation_block = activation + block * q8_block_bytes;
        const float scale = block_scale(weight_block) * block_scale(activation_block);
        int dot = 0;
#pragma unroll
        for (int i = 0; i < q8_block_width; i += 4)
            dot = __dp4a(load_i32(weight_block + 2 + i),
                load_i32(activation_block + 2 + i), dot);
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
    constexpr int ints_per_block = q8_block_width / 4;
    constexpr int lanes_per_block = ints_per_block / 2;
    constexpr int blocks_per_wave = 128 / lanes_per_block;
    for (int block = tid / lanes_per_block; block < blocks; block += blocks_per_wave) {
        const int8_t *weight_block = row_weights + block * q8_block_bytes;
        const int8_t *activation_block = activation + block * q8_block_bytes;
        const float scale = block_scale(weight_block) * block_scale(activation_block);
        const int first_int = (tid % lanes_per_block) * 2;
        int dot = 0;
#pragma unroll 2
        for (int i = 0; i < 2; ++i) {
            const int offset = 2 + 4 * (first_int + i);
            dot = __dp4a(load_i32(weight_block + offset),
                load_i32(activation_block + offset), dot);
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
    const bool four_warps = weights->blocks >= 128;
    if (four_warps)
        q8_gemv_four_warp<<<weights->rows, 128, 0, stream>>>(weights->device_data,
            activation, weights->rows, weights->blocks, output);
    else
        q8_gemv_one_warp<<<weights->rows, 32, 0, stream>>>(weights->device_data,
            activation, weights->rows, weights->blocks, output);
}

} // namespace

struct fortai_cuda_q8_context {
    fortai_cuda_q8_context_impl impl;
};

struct fortai_cuda_q8_weights {
    fortai_cuda_q8_weights_impl impl;
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
    *context = created;
    return FORTAI_CUDA_OK;
}

extern "C" int fortai_cuda_q8_context_destroy(fortai_cuda_q8_context *context) {
    if (!context) return FORTAI_CUDA_OK;
    cudaSetDevice(context->impl.device);
    cudaEventDestroy(context->impl.start);
    cudaEventDestroy(context->impl.stop);
    cudaStreamDestroy(context->impl.stream);
    delete context;
    return FORTAI_CUDA_OK;
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
    cudaError_t error = cudaEventRecord(context->impl.start, context->impl.stream);
    if (error == cudaSuccess) {
        launch_q8(const_cast<fortai_cuda_q8_weights_impl *>(&weights->impl),
            static_cast<const int8_t *>(device_activation),
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
    int8_t *device_activation = nullptr;
    float *device_output = nullptr;
    cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&device_activation),
        static_cast<size_t>(weights->impl.blocks) * q8_block_bytes);
    if (error == cudaSuccess)
        error = cudaMalloc(reinterpret_cast<void **>(&device_output), output_size);
    if (error == cudaSuccess) error = cudaEventRecord(context->impl.start, context->impl.stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(device_activation, host_activation,
            static_cast<size_t>(weights->impl.blocks) * q8_block_bytes,
            cudaMemcpyHostToDevice, context->impl.stream);
    if (error == cudaSuccess) {
        launch_q8(const_cast<fortai_cuda_q8_weights_impl *>(&weights->impl),
            device_activation, device_output, context->impl.stream);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(host_output, device_output, output_size,
            cudaMemcpyDeviceToHost, context->impl.stream);
    if (error == cudaSuccess) error = cudaEventRecord(context->impl.stop, context->impl.stream);
    if (error == cudaSuccess) error = cudaEventSynchronize(context->impl.stop);
    if (error == cudaSuccess) error = cudaEventElapsedTime(elapsed_ms,
        context->impl.start, context->impl.stop);
    cudaFree(device_activation);
    cudaFree(device_output);
    return error == cudaSuccess ? FORTAI_CUDA_OK :
        fail(&context->impl, FORTAI_CUDA_RUNTIME_ERROR, "host matvec", error);
}

extern "C" const char *fortai_cuda_q8_last_error(
    const fortai_cuda_q8_context *context) {
    return context ? context->impl.error : "invalid CUDA context";
}
