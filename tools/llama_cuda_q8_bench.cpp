#include "../backend/cuda/fortai_q8_bench_spec.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace fortai_q8_bench;

int main(int argc, char **argv) {
    options_t options;
    if (!parse_options(argc, argv, options)) return EXIT_FAILURE;
    ggml_backend_t backend = ggml_backend_cuda_init(options.device);
    if (!backend) return EXIT_FAILURE;
    const int blocks = options.width / q8_block_width;
    const size_t weight_bytes = static_cast<size_t>(options.rows) * blocks * q8_block_bytes;
    const size_t activation_bytes = static_cast<size_t>(blocks) * q8_block_bytes;
    std::vector<uint8_t> weights(weight_bytes), activation(activation_bytes);
    std::vector<float> activation_f32(options.width);
    std::vector<float> output(options.rows), oracle(options.rows);
    fill_q8(weights, options.rows, blocks, options.seed, true);
    fill_q8(activation, 1, blocks, options.seed ^ 0x9e3779b9u, false);
    for (int block = 0; block < blocks; ++block) {
        activation[block * q8_block_bytes + 2] = static_cast<uint8_t>(127);
        activation[block * q8_block_bytes + 3] = static_cast<uint8_t>(-127);
    }
    make_oracle(weights, activation, options.rows, blocks, oracle);
    for (int block = 0; block < blocks; ++block) {
        const size_t offset = static_cast<size_t>(block) * q8_block_bytes;
        uint16_t scale_bits;
        std::memcpy(&scale_bits, activation.data() + offset, sizeof(scale_bits));
        for (int i = 0; i < q8_block_width; ++i)
            activation_f32[block * q8_block_width + i] = half_to_float(scale_bits) *
                static_cast<float>(static_cast<int8_t>(activation[offset + 2 + i]));
    }

    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context *ctx = ggml_init(params);
    if (!ctx) return EXIT_FAILURE;
    ggml_tensor *weight_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, options.width, options.rows);
    ggml_tensor *activation_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, options.width, 1);
    ggml_tensor *output_tensor = ggml_mul_mat(ctx, weight_tensor, activation_tensor);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output_tensor);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return EXIT_FAILURE;
    ggml_backend_tensor_set(weight_tensor, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(activation_tensor, activation_f32.data(), 0, activation_f32.size() * sizeof(float));
    for (int i = 0; i < options.warmup; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return EXIT_FAILURE;
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < options.iterations; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return EXIT_FAILURE;
    }
    ggml_backend_synchronize(backend);
    const auto stop = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(output_tensor, output.data(), 0, output.size() * sizeof(float));

    const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int row = 0; row < options.rows; ++row) {
        const float absolute = std::fabs(output[row] - oracle[row]);
        max_abs = std::max(max_abs, absolute);
        max_rel = std::max(max_rel, absolute / std::max(1.0f, std::fabs(oracle[row])));
    }
    const bool correct = max_abs <= 2.0e-3f && max_rel <= 1.0e-4f;
    const double tokens_per_second = options.iterations / (elapsed_ms / 1000.0);
    const double bandwidth = static_cast<double>(weight_bytes + activation_f32.size() * sizeof(float)) *
        tokens_per_second / (1ull << 30);
    std::printf("{\"implementation\":\"llama.cpp-ggml-cuda-q8-weight-f32-activation\",\"device\":%d,\"rows\":%d,"
        "\"width\":%d,\"blocks\":%d,\"iterations\":%d,\"warmup\":%d,\"kernel_ms\":%.6f,"
        "\"tokens_per_second\":%.6f,\"bandwidth_gib_per_second\":%.6f,\"max_abs_error\":%.9g,"
        "\"max_rel_error\":%.9g,\"correct\":%s}\n", options.device, options.rows, options.width, blocks,
        options.iterations, options.warmup, elapsed_ms / options.iterations, tokens_per_second, bandwidth,
        max_abs, max_rel, correct ? "true" : "false");
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
