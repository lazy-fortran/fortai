#include "../backend/cuda/fortai_cuda_backend.h"
#include "../backend/cuda/fortai_q8_bench_spec.h"

#include <algorithm>
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace fortai_q8_bench;

namespace {

float q8_dot(const std::vector<uint8_t> &weights, int row, int blocks,
    const std::vector<uint8_t> &activation) {
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const size_t wo = (static_cast<size_t>(row) * blocks + block) * q8_block_bytes;
        const size_t ao = static_cast<size_t>(block) * q8_block_bytes;
        uint16_t ws = 0, as = 0;
        std::memcpy(&ws, weights.data() + wo, sizeof(ws));
        std::memcpy(&as, activation.data() + ao, sizeof(as));
        int dot = 0;
        for (int i = 0; i < q8_block_width; ++i)
            dot += static_cast<int>(static_cast<int8_t>(weights[wo + 2 + i])) *
                static_cast<int>(static_cast<int8_t>(activation[ao + 2 + i]));
        result += half_to_float(ws) * half_to_float(as) * static_cast<float>(dot);
    }
    return result;
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

void quantize_q8(const std::vector<float> &input, int blocks,
    std::vector<uint8_t> &output) {
    output.assign(static_cast<size_t>(blocks) * q8_block_bytes, 0);
    for (int block = 0; block < blocks; ++block) {
        const int base = block * q8_block_width;
        float maximum = 0.0f;
        for (int i = 0; i < q8_block_width; ++i)
            maximum = std::max(maximum, std::fabs(input[base + i]));
        const float scale = maximum / 127.0f;
        const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
        const uint16_t scale_bits = float_to_half(scale);
        const size_t offset = static_cast<size_t>(block) * q8_block_bytes;
        std::memcpy(output.data() + offset, &scale_bits, sizeof(scale_bits));
        for (int i = 0; i < q8_block_width; ++i)
            output[offset + 2 + i] = static_cast<uint8_t>(static_cast<int8_t>(
                std::lrint(input[base + i] * inverse)));
    }
}

bool check(int code, const fortai_cuda_q8_context *context, const char *operation) {
    if (code == FORTAI_CUDA_OK) return true;
    std::fprintf(stderr, "%s failed (%d): %s\n", operation, code,
        fortai_cuda_q8_last_error(context));
    return false;
}

bool run_case(int device, int hidden, int intermediate) {
    const int input_blocks = hidden / q8_block_width;
    const int output_blocks = intermediate / q8_block_width;
    const size_t gate_bytes = static_cast<size_t>(intermediate) * input_blocks * q8_block_bytes;
    const size_t down_bytes = static_cast<size_t>(hidden) * output_blocks * q8_block_bytes;
    std::vector<uint8_t> gate(gate_bytes), up(gate_bytes), down(down_bytes), activation(
        static_cast<size_t>(input_blocks) * q8_block_bytes);
    fill_q8(gate, intermediate, input_blocks, 0x13579bdfu + intermediate, true);
    fill_q8(up, intermediate, input_blocks, 0x2468ace0u + intermediate, true);
    fill_q8(down, hidden, output_blocks, 0x31415926u + intermediate, true);
    fill_q8(activation, 1, input_blocks, 0x27182818u + intermediate, false);

    std::vector<float> intermediate_oracle(intermediate);
    for (int row = 0; row < intermediate; ++row)
        intermediate_oracle[row] = silu(q8_dot(gate, row, input_blocks, activation)) *
            q8_dot(up, row, input_blocks, activation);
    std::vector<uint8_t> intermediate_q8;
    quantize_q8(intermediate_oracle, output_blocks, intermediate_q8);
    std::vector<float> oracle(hidden);
    for (int row = 0; row < hidden; ++row)
        oracle[row] = q8_dot(down, row, output_blocks, intermediate_q8);

    fortai_cuda_q8_context *context = nullptr;
    if (!check(fortai_cuda_q8_context_create(device, &context), context, "context create"))
        return false;
    fortai_cuda_q8_weights *gate_device = nullptr, *up_device = nullptr, *down_device = nullptr;
    bool ok = check(fortai_cuda_q8_weights_upload(context, gate.data(), gate.size(), intermediate,
        hidden, &gate_device), context, "gate upload") &&
        check(fortai_cuda_q8_weights_upload(context, up.data(), up.size(), intermediate, hidden,
            &up_device), context, "up upload") &&
        check(fortai_cuda_q8_weights_upload(context, down.data(), down.size(), hidden, intermediate,
            &down_device), context, "down upload");
    std::vector<float> actual(hidden);
    float elapsed_ms = 0.0f;
    if (ok) ok = check(fortai_cuda_q8_ffn_host(context, gate_device, up_device, down_device,
        activation.data(), activation.size(), actual.data(), actual.size() * sizeof(float),
        &elapsed_ms), context, "ffn");
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int i = 0; i < hidden; ++i) {
        const float error = std::fabs(actual[i] - oracle[i]);
        max_abs = std::max(max_abs, error);
        const float relative = error / std::max(1.0f, std::fabs(oracle[i]));
        max_rel = std::max(max_rel, relative);
    }
    std::printf("case_hidden=%d case_intermediate=%d case_blocks=%d elapsed_ms=%.6f "
        "max_abs=%.9g max_rel=%.9g\n", hidden, intermediate, output_blocks, elapsed_ms,
        max_abs, max_rel);
    if (gate_device) ok = check(fortai_cuda_q8_weights_destroy(gate_device), context,
        "gate free") && ok;
    if (up_device) ok = check(fortai_cuda_q8_weights_destroy(up_device), context,
        "up free") && ok;
    if (down_device) ok = check(fortai_cuda_q8_weights_destroy(down_device), context,
        "down free") && ok;
    ok = check(fortai_cuda_q8_context_destroy(context), nullptr, "context destroy") && ok;
    /* The final down projection consumes a GPU fast-math SiLU result that is
     * re-quantized to Q8.  A small SiLU ulp difference can change one integer
     * code, so validate the independent end-to-end oracle at 10% relative
     * error while still rejecting lane-mapping and layout failures. */
    return ok && std::isfinite(max_abs) && std::isfinite(max_rel) && max_rel <= 0.1f;
}

} // namespace

int main(int argc, char **argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [CUDA_DEVICE]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = argc == 2 ? std::atoi(argv[1]) : 0;
    std::fesetround(FE_TONEAREST);
    const bool fused = run_case(device, 1024, 3584);
    const bool fallback = run_case(device, 1024, 4096);
    return fused && fallback ? EXIT_SUCCESS : EXIT_FAILURE;
}
