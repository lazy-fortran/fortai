#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace fortai_q8_bench {

constexpr int q8_block_width = 32;
constexpr int q8_block_bytes = 34;

inline uint32_t next_random(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t fraction = bits & 0x007fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        fraction = (fraction | 0x00800000u) >> static_cast<uint32_t>(1 - exponent);
        if (bits & 0x00001000u) fraction += 0x00002000u;
        return static_cast<uint16_t>(sign | (fraction >> 13));
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (fraction & 0x00001000u) {
        fraction += 0x00002000u;
        if (fraction & 0x00800000u) {
            fraction = 0;
            if (++exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
        (fraction >> 13));
}

inline float half_to_float(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value) & 0x8000u) << 16;
    const uint32_t exponent = (static_cast<uint32_t>(value) >> 10) & 0x1fu;
    const uint32_t fraction = static_cast<uint32_t>(value) & 0x03ffu;
    uint32_t bits;
    if (exponent == 0u) {
        if (fraction == 0u) bits = sign;
        else {
            uint32_t normalized = fraction;
            uint32_t shift = 0u;
            while ((normalized & 0x0400u) == 0u) { normalized <<= 1; ++shift; }
            bits = sign | (static_cast<uint32_t>(127 - 15 - shift + 1) << 23) |
                ((normalized & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) bits = sign | 0x7f800000u | (fraction << 13);
    else bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline void fill_q8(std::vector<uint8_t> &data, int rows, int blocks,
    uint32_t seed, bool per_row) {
    for (int row = 0; row < (per_row ? rows : 1); ++row) {
        for (int block = 0; block < blocks; ++block) {
            const size_t offset = (static_cast<size_t>(row) * blocks + block) * q8_block_bytes;
            const float scale = 0.004f + 0.012f * static_cast<float>(next_random(seed) & 1023u) / 1023.0f;
            const uint16_t scale_bits = float_to_half(scale);
            std::memcpy(data.data() + offset, &scale_bits, sizeof(scale_bits));
            for (int i = 0; i < q8_block_width; ++i)
                data[offset + 2 + i] = static_cast<uint8_t>(static_cast<int>(next_random(seed) % 255u) - 127);
        }
    }
}

inline void make_oracle(const std::vector<uint8_t> &weights,
    const std::vector<uint8_t> &activation, int rows, int blocks,
    std::vector<float> &oracle) {
    for (int row = 0; row < rows; ++row) {
        float result = 0.0f;
        for (int block = 0; block < blocks; ++block) {
            const size_t wo = (static_cast<size_t>(row) * blocks + block) * q8_block_bytes;
            const size_t ao = static_cast<size_t>(block) * q8_block_bytes;
            uint16_t ws, as;
            std::memcpy(&ws, weights.data() + wo, sizeof(ws));
            std::memcpy(&as, activation.data() + ao, sizeof(as));
            int dot = 0;
            for (int i = 0; i < q8_block_width; ++i)
                dot += static_cast<int>(static_cast<int8_t>(weights[wo + 2 + i])) *
                    static_cast<int>(static_cast<int8_t>(activation[ao + 2 + i]));
            result += half_to_float(ws) * half_to_float(as) * static_cast<float>(dot);
        }
        oracle[row] = result;
    }
}

struct options_t {
    int device = 0;
    int rows = 4096;
    int width = 4096;
    int iterations = 1000;
    int warmup = 100;
    uint32_t seed = 0x6f727461u;
};

inline bool parse_int(const char *text, int &value) {
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end || parsed < 1 || parsed > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(parsed);
    return true;
}

inline bool parse_device(const char *text, int &value) {
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end || parsed < 0 || parsed > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(parsed);
    return true;
}

inline bool parse_options(int argc, char **argv, options_t &options) {
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return false;
        int *target = nullptr;
        if (!std::strcmp(argv[i], "--device")) {
            if (!parse_device(argv[++i], options.device)) return false;
            continue;
        } else if (!std::strcmp(argv[i], "--rows")) target = &options.rows;
        else if (!std::strcmp(argv[i], "--width")) target = &options.width;
        else if (!std::strcmp(argv[i], "--iterations")) target = &options.iterations;
        else if (!std::strcmp(argv[i], "--warmup")) target = &options.warmup;
        if (target) {
            if (!parse_int(argv[++i], *target)) return false;
        } else if (!std::strcmp(argv[i], "--seed")) {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(argv[++i], &end, 0);
            if (end == argv[i] || *end || parsed > std::numeric_limits<uint32_t>::max()) return false;
            options.seed = static_cast<uint32_t>(parsed);
        } else return false;
    }
    return options.width % q8_block_width == 0;
}

} // namespace fortai_q8_bench
