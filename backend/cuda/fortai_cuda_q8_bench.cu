#include "fortai_q8_bench_spec.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace fortai_q8_bench;

static void check_cuda(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ int load_i32(const int8_t *address) {
    const uint16_t *values = reinterpret_cast<const uint16_t *>(address);
    return static_cast<int>(values[0]) | (static_cast<int>(values[1]) << 16);
}

__global__ void fortai_q8_gemv_kernel(const int8_t *__restrict__ weights,
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
        const uint16_t ws = *reinterpret_cast<const uint16_t *>(weight_block);
        const uint16_t as = *reinterpret_cast<const uint16_t *>(activation_block);
        const float scale = __half2float(*reinterpret_cast<const __half *>(&ws)) *
            __half2float(*reinterpret_cast<const __half *>(&as));
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

__global__ void fortai_q8_gemv_4warp_kernel(const int8_t *__restrict__ weights,
    const int8_t *__restrict__ activation, int rows, int blocks,
    float *__restrict__ output) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (row >= rows) return;
    const int8_t *row_weights = weights + static_cast<size_t>(row) * blocks * q8_block_bytes;
    float accumulator = 0.0f;
    // Match llama.cpp's Q8_0 x Q8_1 MMVQ work decomposition: four lanes
    // cooperate on one 32-value block, each lane issuing two DP4A ops.
    constexpr int ints_per_block = q8_block_width / 4;
    constexpr int lanes_per_block = ints_per_block / 2;
    constexpr int blocks_per_wave = 128 / lanes_per_block;
    for (int block = tid / lanes_per_block; block < blocks; block += blocks_per_wave) {
        const int8_t *weight_block = row_weights + block * q8_block_bytes;
        const int8_t *activation_block = activation + block * q8_block_bytes;
        const uint16_t ws = *reinterpret_cast<const uint16_t *>(weight_block);
        const uint16_t as = *reinterpret_cast<const uint16_t *>(activation_block);
        const float scale = __half2float(*reinterpret_cast<const __half *>(&ws)) *
            __half2float(*reinterpret_cast<const __half *>(&as));
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

int main(int argc, char **argv) {
    options_t options;
    if (!parse_options(argc, argv, options)) return EXIT_FAILURE;
    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, options.device), "cudaGetDeviceProperties");
    const int blocks = options.width / q8_block_width;
    const size_t weight_bytes = static_cast<size_t>(options.rows) * blocks * q8_block_bytes;
    const size_t activation_bytes = static_cast<size_t>(blocks) * q8_block_bytes;
    std::vector<uint8_t> weights(weight_bytes), activation(activation_bytes);
    std::vector<float> output(options.rows), oracle(options.rows);
    fill_q8(weights, options.rows, blocks, options.seed, true);
    fill_q8(activation, 1, blocks, options.seed ^ 0x9e3779b9u, false);
    for (int block = 0; block < blocks; ++block) {
        activation[block * q8_block_bytes + 2] = static_cast<uint8_t>(127);
        activation[block * q8_block_bytes + 3] = static_cast<uint8_t>(-127);
    }
    make_oracle(weights, activation, options.rows, blocks, oracle);
    int8_t *device_weights = nullptr, *device_activation = nullptr;
    float *device_output = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_weights), weight_bytes), "cudaMalloc weights");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_activation), activation_bytes), "cudaMalloc activation");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_output), options.rows * sizeof(float)), "cudaMalloc output");
    check_cuda(cudaMemcpy(device_weights, weights.data(), weight_bytes, cudaMemcpyHostToDevice), "copy weights");
    check_cuda(cudaMemcpy(device_activation, activation.data(), activation_bytes, cudaMemcpyHostToDevice), "copy activation");
    // Four warps amortize the long-vector reduction, but their shared partial
    // reduction costs more than they save at the 128-block crossover.
    const bool use_four_warps = blocks >= 128;
    const int threads = use_four_warps ? 128 : 32;
    for (int i = 0; i < options.warmup; ++i) {
        if (use_four_warps)
            fortai_q8_gemv_4warp_kernel<<<options.rows, threads>>>(device_weights, device_activation, options.rows, blocks, device_output);
        else
            fortai_q8_gemv_kernel<<<options.rows, threads>>>(device_weights, device_activation, options.rows, blocks, device_output);
    }
    check_cuda(cudaGetLastError(), "warmup kernel");
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "event start");
    check_cuda(cudaEventCreate(&stop), "event stop");
    check_cuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < options.iterations; ++i) {
        if (use_four_warps)
            fortai_q8_gemv_4warp_kernel<<<options.rows, threads>>>(device_weights, device_activation, options.rows, blocks, device_output);
        else
            fortai_q8_gemv_kernel<<<options.rows, threads>>>(device_weights, device_activation, options.rows, blocks, device_output);
    }
    check_cuda(cudaGetLastError(), "timed kernel");
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "timed synchronize");
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
    check_cuda(cudaMemcpy(output.data(), device_output, options.rows * sizeof(float), cudaMemcpyDeviceToHost), "copy output");
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int row = 0; row < options.rows; ++row) {
        const float absolute = std::fabs(output[row] - oracle[row]);
        max_abs = std::max(max_abs, absolute);
        max_rel = std::max(max_rel, absolute / std::max(1.0f, std::fabs(oracle[row])));
    }
    const bool correct = max_abs <= 2.0e-3f && max_rel <= 1.0e-4f;
    const double tokens_per_second = options.iterations / (elapsed_ms / 1000.0);
    const double bandwidth = static_cast<double>(weight_bytes + activation_bytes) * tokens_per_second / (1ull << 30);
    std::printf("{\"implementation\":\"fortai-cuda-q8\",\"device\":%d,\"device_name\":\"%s\","
        "\"compute_capability\":\"%d.%d\",\"rows\":%d,\"width\":%d,\"blocks\":%d,"
        "\"iterations\":%d,\"warmup\":%d,\"kernel_ms\":%.6f,\"tokens_per_second\":%.6f,"
        "\"bandwidth_gib_per_second\":%.6f,\"max_abs_error\":%.9g,\"max_rel_error\":%.9g,\"correct\":%s}\n",
        options.device, properties.name, properties.major, properties.minor, options.rows, options.width, blocks,
        options.iterations, options.warmup, elapsed_ms / options.iterations, tokens_per_second, bandwidth,
        max_abs, max_rel, correct ? "true" : "false");
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(device_weights); cudaFree(device_activation); cudaFree(device_output);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
