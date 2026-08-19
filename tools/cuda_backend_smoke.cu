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
    for (int row = 0; row < rows; ++row) {
        const float host_error = std::fabs(host_output[row] - oracle[row]);
        const float resident_error = std::fabs(resident_output[row] - oracle[row]);
        host_abs = std::max(host_abs, host_error);
        resident_abs = std::max(resident_abs, resident_error);
        host_rel = std::max(host_rel, host_error / std::max(1.0f, std::fabs(oracle[row])));
        resident_rel = std::max(resident_rel, resident_error / std::max(1.0f, std::fabs(oracle[row])));
    }
    const bool correct = host_abs <= 2.0e-3f && host_rel <= 1.0e-4f &&
        resident_abs <= 2.0e-3f && resident_rel <= 1.0e-4f;
    std::printf("{\"implementation\":\"fortai-cuda-q8-backend-abi\",\"device\":%d,"
        "\"rows\":%d,\"width\":%d,\"host_elapsed_ms\":%.6f,"
        "\"resident_kernel_ms\":%.6f,\"host_max_abs_error\":%.9g,"
        "\"host_max_rel_error\":%.9g,\"resident_max_abs_error\":%.9g,"
        "\"resident_max_rel_error\":%.9g,\"correct\":%s}\n", device, rows, width,
        host_ms, resident_ms, host_abs, host_rel, resident_abs, resident_rel,
        correct ? "true" : "false");

    check(fortai_cuda_q8_device_buffer_destroy(context, device_output), context, "output free");
    check(fortai_cuda_q8_device_buffer_destroy(context, device_activation), context, "activation free");
    check(fortai_cuda_q8_weights_destroy(device_weights), context, "weight free");
    check(fortai_cuda_q8_context_destroy(context), nullptr, "context destroy");
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
