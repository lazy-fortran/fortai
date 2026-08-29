#include "../backend/cuda/fortai_cuda_backend.h"
#include "../backend/cuda/fortai_q8_bench_spec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace fortai_q8_bench;

static int argument_device(int argc, char **argv) {
    if (argc == 1) return 0;
    if (argc == 3 && std::string(argv[1]) == "--device") {
        char *end = nullptr;
        const long value = std::strtol(argv[2], &end, 10);
        if (end != argv[2] && *end == 0 && value >= 0 && value <= 64) return static_cast<int>(value);
    }
    std::fprintf(stderr, "usage: %s [--device INDEX]\n", argv[0]);
    std::exit(EXIT_FAILURE);
}

static void check(int code, const fortai_cuda_q8_context *context, const char *operation) {
    if (code == FORTAI_CUDA_OK) return;
    std::fprintf(stderr, "%s failed (%d): %s\n", operation, code,
        fortai_cuda_q8_last_error(context));
    std::exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    constexpr int rows = 4096;
    constexpr int width = 4096;
    constexpr int blocks = width / q8_block_width;
    const int device = argument_device(argc, argv);
    const size_t weight_bytes = static_cast<size_t>(rows) * blocks * q8_block_bytes;
    const size_t activation_bytes = static_cast<size_t>(blocks) * q8_block_bytes;
    std::vector<uint8_t> weights(weight_bytes), activation(activation_bytes);
    std::vector<float> oracle(rows), host_output(rows), resident_output(rows);
    std::vector<float> pair_first(rows), pair_second(rows);
    std::vector<float> triplet_first(rows), triplet_second(rows), triplet_third(rows);
    fill_q8(weights, rows, blocks, 0x6f727461u, true);
    fill_q8(activation, 1, blocks, 0x6f727461u ^ 0x9e3779b9u, false);
    for (int block = 0; block < blocks; ++block) {
        activation[block * q8_block_bytes + 2] = static_cast<uint8_t>(127);
        activation[block * q8_block_bytes + 3] = static_cast<uint8_t>(-127);
    }
    make_oracle(weights, activation, rows, blocks, oracle);

    fortai_cuda_q8_context *context = nullptr;
    check(fortai_cuda_q8_context_create(device, &context), context, "context create");
    fortai_cuda_q8_weights *device_weights = nullptr;
    check(fortai_cuda_q8_weights_upload(context, weights.data(), weights.size(), rows, width,
        &device_weights), context, "weight upload");

    float host_ms = 0.0f;
    check(fortai_cuda_q8_matvec_host(context, device_weights, activation.data(), activation.size(),
        host_output.data(), host_output.size() * sizeof(float), &host_ms), context, "host matvec");
    float pair_ms = 0.0f;
    check(fortai_cuda_q8_matvec_host_pair(context, device_weights, device_weights,
        activation.data(), activation.size(), pair_first.data(), pair_first.size() * sizeof(float),
        pair_second.data(), pair_second.size() * sizeof(float), &pair_ms), context, "host pair");
    float triplet_ms = 0.0f;
    check(fortai_cuda_q8_matvec_host_triplet(context, device_weights, device_weights, device_weights,
        activation.data(), activation.size(), triplet_first.data(), triplet_first.size() * sizeof(float),
        triplet_second.data(), triplet_second.size() * sizeof(float), triplet_third.data(),
        triplet_third.size() * sizeof(float), &triplet_ms), context, "host triplet");

    void *device_activation = nullptr;
    void *device_output = nullptr;
    check(fortai_cuda_q8_device_buffer_create(context, activation.size(), &device_activation),
        context, "activation allocation");
    check(fortai_cuda_q8_device_buffer_create(context, resident_output.size() * sizeof(float),
        &device_output), context, "output allocation");
    check(fortai_cuda_q8_device_buffer_upload(context, device_activation, activation.data(),
        activation.size()), context, "activation upload");
    float resident_ms = 0.0f;
    check(fortai_cuda_q8_matvec_resident(context, device_weights, device_activation,
        device_output, &resident_ms), context, "resident matvec");
    check(fortai_cuda_q8_device_buffer_download(context, resident_output.data(), device_output,
        resident_output.size() * sizeof(float)), context, "output download");

    float host_abs = 0.0f, host_rel = 0.0f, resident_abs = 0.0f, resident_rel = 0.0f;
    float grouped_abs = 0.0f, grouped_rel = 0.0f;
    for (int row = 0; row < rows; ++row) {
        const float host_error = std::fabs(host_output[row] - oracle[row]);
        const float resident_error = std::fabs(resident_output[row] - oracle[row]);
        host_abs = std::max(host_abs, host_error);
        resident_abs = std::max(resident_abs, resident_error);
        host_rel = std::max(host_rel, host_error / std::max(1.0f, std::fabs(oracle[row])));
        resident_rel = std::max(resident_rel, resident_error / std::max(1.0f, std::fabs(oracle[row])));
        const float grouped_values[] = {
            pair_first[row], pair_second[row], triplet_first[row], triplet_second[row], triplet_third[row]
        };
        for (const float value : grouped_values) {
            const float error = std::fabs(value - oracle[row]);
            grouped_abs = std::max(grouped_abs, error);
            grouped_rel = std::max(grouped_rel, error / std::max(1.0f, std::fabs(oracle[row])));
        }
    }
    const bool correct = host_abs <= 2.0e-3f && host_rel <= 1.0e-4f &&
        resident_abs <= 2.0e-3f && resident_rel <= 1.0e-4f &&
        grouped_abs <= 2.0e-3f && grouped_rel <= 1.0e-4f;

    constexpr int topk_rows = 3;
    constexpr int topk_width = 248320;
    constexpr int topk_count = 20;
    std::vector<float> topk_input(topk_rows * topk_width);
    std::vector<int> topk_indices(topk_rows * topk_count);
    std::vector<float> topk_values(topk_rows * topk_count);
    for (int row = 0; row < topk_rows; ++row) {
        for (int column = 0; column < topk_width; ++column) {
            const uint32_t mixed = static_cast<uint32_t>(column * 2654435761u) ^
                static_cast<uint32_t>((row + 1) * 2246822519u);
            topk_input[row * topk_width + column] =
                static_cast<float>(static_cast<int32_t>(mixed)) / 2147483648.0f;
        }
        topk_input[row * topk_width + 17] = 3.0f;
        topk_input[row * topk_width + 29] = 3.0f;
    }
    void *device_topk = nullptr;
    check(fortai_cuda_q8_device_buffer_create(context,
        topk_input.size() * sizeof(float), &device_topk), context,
        "top-k allocation");
    check(fortai_cuda_q8_device_buffer_upload(context, device_topk, topk_input.data(),
        topk_input.size() * sizeof(float)), context, "top-k input upload");
    check(fortai_cuda_qwen35_topk_rows_device(context, device_topk, topk_width,
        topk_rows, topk_count, topk_indices.data(), topk_values.data()), context,
        "row top-k");
    bool topk_correct = true;
    for (int row = 0; row < topk_rows; ++row) {
        std::vector<int> oracle_indices(topk_width);
        for (int column = 0; column < topk_width; ++column) oracle_indices[column] = column;
        std::partial_sort(oracle_indices.begin(), oracle_indices.begin() + topk_count,
            oracle_indices.end(), [&](int left, int right) {
                const float left_value = topk_input[row * topk_width + left];
                const float right_value = topk_input[row * topk_width + right];
                return left_value > right_value ||
                    (left_value == right_value && left < right);
            });
        for (int rank = 0; rank < topk_count; ++rank) {
            const int offset = row * topk_count + rank;
            const int oracle_index = oracle_indices[rank];
            if (topk_indices[offset] != oracle_index ||
                topk_values[offset] != topk_input[row * topk_width + oracle_index]) {
                topk_correct = false;
            }
        }
    }
    std::printf("{\"implementation\":\"fortai-cuda-q8-backend-abi\",\"device\":%d,"
        "\"rows\":%d,\"width\":%d,\"host_elapsed_ms\":%.6f,"
        "\"resident_kernel_ms\":%.6f,\"grouped_pair_ms\":%.6f,\"grouped_triplet_ms\":%.6f,"
        "\"host_max_abs_error\":%.9g,"
        "\"host_max_rel_error\":%.9g,\"resident_max_abs_error\":%.9g,"
        "\"resident_max_rel_error\":%.9g,\"grouped_max_abs_error\":%.9g,"
        "\"grouped_max_rel_error\":%.9g,\"topk_correct\":%s,\"correct\":%s}\n", device, rows, width,
        host_ms, resident_ms, pair_ms, triplet_ms, host_abs, host_rel, resident_abs, resident_rel,
        grouped_abs, grouped_rel,
        topk_correct ? "true" : "false", correct && topk_correct ? "true" : "false");

    check(fortai_cuda_q8_device_buffer_destroy(context, device_topk), context, "top-k free");
    check(fortai_cuda_q8_device_buffer_destroy(context, device_output), context, "output free");
    check(fortai_cuda_q8_device_buffer_destroy(context, device_activation), context, "activation free");
    check(fortai_cuda_q8_weights_destroy(device_weights), context, "weight free");
    check(fortai_cuda_q8_context_destroy(context), nullptr, "context destroy");
    return correct && topk_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
