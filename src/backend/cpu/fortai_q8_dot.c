#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#if defined(__GNUC__)
#pragma GCC optimize("O3")
#endif

__attribute__((target("avx2,fma")))
static inline __m256 expf_avx2(__m256 value)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 max_input = _mm256_set1_ps(88.3762626647949f);
    const __m256 min_input = _mm256_set1_ps(-88.3762626647949f);
    const __m256 c1 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 c2 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 c3 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 c4 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 c5 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 c6 = _mm256_set1_ps(5.0000001201e-1f);
    __m256 fx, truncated, z, polynomial;
    __m256i exponent;

    value = _mm256_min_ps(value, max_input);
    value = _mm256_max_ps(value, min_input);
    fx = _mm256_add_ps(_mm256_mul_ps(value, log2e), half);
    exponent = _mm256_cvttps_epi32(fx);
    truncated = _mm256_cvtepi32_ps(exponent);
    exponent = _mm256_sub_epi32(exponent, _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(truncated, fx, _CMP_GT_OQ)),
        _mm256_set1_epi32(1)));
    fx = _mm256_cvtepi32_ps(exponent);
    value = _mm256_sub_ps(value, _mm256_mul_ps(fx, ln2_hi));
    value = _mm256_sub_ps(value, _mm256_mul_ps(fx, ln2_lo));
    z = _mm256_mul_ps(value, value);
    polynomial = c1;
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, value), c2);
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, value), c3);
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, value), c4);
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, value), c5);
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, value), c6);
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, z), value);
    polynomial = _mm256_add_ps(polynomial, one);
    exponent = _mm256_add_epi32(exponent, _mm256_set1_epi32(0x7f));
    exponent = _mm256_slli_epi32(exponent, 23);
    return _mm256_mul_ps(polynomial, _mm256_castsi256_ps(exponent));
}

static inline float half_to_float(uint16_t value)
{
    const uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    const uint32_t exponent = ((uint32_t)value >> 10) & 0x1fu;
    const uint32_t fraction = (uint32_t)value & 0x03ffu;
    uint32_t bits;

    if (exponent == 0u) {
        if (fraction == 0u) {
            bits = sign;
        } else {
            uint32_t normalized = fraction;
            uint32_t shift = 0u;
            while ((normalized & 0x0400u) == 0u) {
                normalized <<= 1;
                ++shift;
            }
            bits = sign | ((uint32_t)(127 - 15 - shift + 1) << 23) |
                ((normalized & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    {
        float result;
        __builtin_memcpy(&result, &bits, sizeof(result));
        return result;
    }
}

static inline uint16_t float_to_half(float value)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t fraction = bits & 0x007fffffu;

    if (exponent <= 0) {
        if (exponent < -10)
            return (uint16_t)sign;
        fraction = (fraction | 0x00800000u) >> (uint32_t)(1 - exponent);
        if ((bits & 0x00001000u) != 0u)
            fraction += 0x00002000u;
        return (uint16_t)(sign | (fraction >> 13));
    }
    if (exponent >= 31)
        return (uint16_t)(sign | 0x7c00u);
    if ((fraction & 0x00001000u) != 0u) {
        fraction += 0x00002000u;
        if ((fraction & 0x00800000u) != 0u) {
            fraction = 0;
            ++exponent;
            if (exponent >= 31)
                return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (fraction >> 13));
}

__attribute__((always_inline))
static inline uint16_t load_u16(const int8_t *address)
{
    uint16_t value;
    __builtin_memcpy(&value, address, sizeof(value));
    return value;
}

uint16_t fortai_float_to_half(float value)
{
    return float_to_half(value);
}

static void q8_quantize_scalar(const float *__restrict input,
    int8_t *__restrict output, float *__restrict scales, int64_t count)
{
    int64_t offset;
    for (offset = 0; offset < count; offset += 32) {
        const int64_t output_offset = (offset / 32) * 34;
        float maximum = 0.0f;
        int i;
        for (i = 0; i < 32; ++i) {
            const float magnitude = fabsf(input[offset + i]);
            if (magnitude > maximum)
                maximum = magnitude;
        }
        const float scale = maximum / 127.0f;
        const float inverse = maximum != 0.0f ? 127.0f / maximum : 0.0f;
        const uint16_t scale_bits = float_to_half(scale);
        scales[offset / 32] = half_to_float(scale_bits);
        __builtin_memcpy(output + output_offset, &scale_bits, sizeof(scale_bits));
        for (i = 0; i < 32; ++i)
            output[output_offset + 2 + i] =
                (int8_t)roundf(input[offset + i] * inverse);
    }
}

__attribute__((target("avx2,f16c")))
static void q8_quantize_avx2(const float *__restrict input,
    int8_t *__restrict output, float *__restrict scales, int64_t count)
{
    int64_t offset;
    for (offset = 0; offset < count; offset += 32) {
        const int64_t output_offset = (offset / 32) * 34;
        const __m256 sign_bit = _mm256_set1_ps(-0.0f);
        __m256 v0 = _mm256_loadu_ps(input + offset);
        __m256 v1 = _mm256_loadu_ps(input + offset + 8);
        __m256 v2 = _mm256_loadu_ps(input + offset + 16);
        __m256 v3 = _mm256_loadu_ps(input + offset + 24);
        __m256 maximum = _mm256_andnot_ps(sign_bit, v0);
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign_bit, v1));
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign_bit, v2));
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign_bit, v3));
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maximum, 1),
            _mm256_castps256_ps128(maximum));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maximum_scalar = _mm_cvtss_f32(max4);
        const float scale = maximum_scalar / 127.0f;
        const float inverse = maximum_scalar != 0.0f ? 127.0f / maximum_scalar : 0.0f;
        const __m256 multiplier = _mm256_set1_ps(inverse);
        const uint16_t scale_bits = _cvtss_sh(scale, _MM_FROUND_TO_NEAREST_INT);
        scales[offset / 32] = _cvtsh_ss(scale_bits);

        const __m256i q0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, multiplier),
            _MM_FROUND_TO_NEAREST_INT));
        const __m256i q1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, multiplier),
            _MM_FROUND_TO_NEAREST_INT));
        const __m256i q2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, multiplier),
            _MM_FROUND_TO_NEAREST_INT));
        const __m256i q3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, multiplier),
            _MM_FROUND_TO_NEAREST_INT));
        /* Match llama.cpp's AVX2 pack/permutation sequence.  The pack
         * instructions operate on independent 128-bit lanes, so the final
         * permutation restores the original q0/q1/q2/q3 element order. */
        __m256i packed01 = _mm256_packs_epi32(q0, q1);
        __m256i packed23 = _mm256_packs_epi32(q2, q3);
        __m256i packed = _mm256_packs_epi16(packed01, packed23);
        const __m256i permutation = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        packed = _mm256_permutevar8x32_epi32(packed, permutation);
        __builtin_memcpy(output + output_offset, &scale_bits, sizeof(scale_bits));
        _mm256_storeu_si256((__m256i *)(output + output_offset + 2), packed);
    }
}

void fortai_q8_quantize(const float *__restrict input,
    int8_t *__restrict output, float *__restrict scales,
    int64_t count)
{
    if (count <= 0 || count % 32 != 0)
        return;
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) {
        q8_quantize_avx2(input, output, scales, count);
        return;
    }
#endif
    q8_quantize_scalar(input, output, scales, count);
}

static void q8_dequantize_row_scalar(const int8_t *__restrict weights,
    float *__restrict output, int64_t block_count)
{
    int64_t block;
    for (block = 0; block < block_count; ++block) {
        const int64_t offset = block * 34;
        const uint16_t scale_bits = load_u16(weights + offset);
        const float scale = half_to_float(scale_bits);
        int i;
        for (i = 0; i < 32; ++i)
            output[block * 32 + i] = scale * (float)weights[offset + 2 + i];
    }
}

__attribute__((target("avx2,f16c,fma")))
static void q8_dequantize_row_avx2(const int8_t *__restrict weights,
    float *__restrict output, int64_t block_count)
{
    int64_t block;
    for (block = 0; block < block_count; ++block) {
        const int64_t input_offset = block * 34;
        const int64_t output_offset = block * 32;
        const __m256 scale = _mm256_set1_ps(_cvtsh_ss(
            load_u16(weights + input_offset)));
        int part;
        for (part = 0; part < 4; ++part) {
            const __m128i q8 = _mm_loadl_epi64((const __m128i *)
                (weights + input_offset + 2 + part * 8));
            const __m256i q32 = _mm256_cvtepi8_epi32(q8);
            const __m256 values = _mm256_mul_ps(scale, _mm256_cvtepi32_ps(q32));
            _mm256_storeu_ps(output + output_offset + part * 8, values);
        }
    }
}

void fortai_q8_dequantize_row(const int8_t *__restrict weights,
    float *__restrict output, int64_t block_count)
{
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) {
        q8_dequantize_row_avx2(weights, output, block_count);
        return;
    }
#endif
    q8_dequantize_row_scalar(weights, output, block_count);
}

static float q8_dot_scalar(const int8_t *__restrict weights,
    const int8_t *__restrict quantized, const float *__restrict scales,
    int64_t row, int64_t block_count)
{
    float result = 0.0f;
    int64_t block;

    for (block = 0; block < block_count; ++block) {
        const int64_t offset = (row * block_count + block) * 34;
        const int64_t activation_offset = block * 34;
        const uint16_t scale_bits = (uint16_t)(uint8_t)weights[offset] |
            ((uint16_t)(uint8_t)weights[offset + 1] << 8);
        const uint16_t activation_scale_bits =
            (uint16_t)(uint8_t)quantized[activation_offset] |
            ((uint16_t)(uint8_t)quantized[activation_offset + 1] << 8);
        int32_t dot = 0;
        int i;

        for (i = 0; i < 32; ++i)
            dot += (int32_t)weights[offset + 2 + i] *
                (int32_t)quantized[activation_offset + 2 + i];
        result += half_to_float(scale_bits) * half_to_float(activation_scale_bits) *
            (float)dot;
    }
    return result;
}

__attribute__((target("avx2,f16c,fma")))
static float q8_dot_avx2(const int8_t *__restrict weights,
    const int8_t *__restrict quantized, const float *__restrict scales,
    int64_t row, int64_t block_count)
{
    __m256 accumulator = _mm256_setzero_ps();
    const int8_t *row_start = weights + row * block_count * 34;
    const int8_t *activation_start = quantized;
    const int32_t blocks = (int32_t)block_count;
    int64_t offset = 0;
    int32_t block;

    for (block = 0; block < blocks; ++block) {
#if defined(__GNUC__)
        /* Keep the integer induction variable live; GCC otherwise folds the
         * block counter into a 64-bit pointer-end comparison. */
        __asm__ volatile("" : "+r"(block));
#endif
        const uint16_t scale_bits = load_u16(row_start + offset);
        const uint16_t activation_scale_bits = load_u16(activation_start + offset);
        const __m256i weight = _mm256_loadu_si256(
            (const __m256i *)(row_start + offset + 2));
        const __m256i activation = _mm256_loadu_si256(
            (const __m256i *)(activation_start + offset + 2));
        const __m256 scale = _mm256_set1_ps(_cvtsh_ss(scale_bits) *
            _cvtsh_ss(activation_scale_bits));
        /* psignb(x, x) is the exact full-width sequence used by llama.cpp
         * for saturating int8 absolute values, without a separate pabsb. */
        const __m256i absolute_weight = _mm256_sign_epi8(weight, weight);
        const __m256i signed_activation = _mm256_sign_epi8(activation, weight);
        const __m256i products = _mm256_maddubs_epi16(absolute_weight,
            signed_activation);
        const __m256i pairs = _mm256_madd_epi16(products,
            _mm256_set1_epi16(1));
        const __m256 dot = _mm256_cvtepi32_ps(pairs);
        accumulator = _mm256_fmadd_ps(scale, dot, accumulator);
        offset += 34;
    }
    {
        const __m128 lower = _mm256_castps256_ps128(accumulator);
        const __m128 upper = _mm256_extractf128_ps(accumulator, 1);
        __m128 sum = _mm_add_ps(lower, upper);
        sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
        return _mm_cvtss_f32(sum);
    }
}

float fortai_q8_dot(const int8_t *__restrict weights,
    const int8_t *__restrict quantized, const float *__restrict scales,
    int64_t row, int64_t block_count)
{
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c"))
        return q8_dot_avx2(weights, quantized, scales, row, block_count);
#endif
    return q8_dot_scalar(weights, quantized, scales, row, block_count);
}

__attribute__((target("avx2,fma")))
static inline float horizontal_sum(__m256 value)
{
    const __m128 lower = _mm256_castps256_ps128(value);
    const __m128 upper = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(lower, upper);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma")))
static void gdn_step_avx2(float *state, const float *key, const float *value,
    const float *query, float decay, float beta, int64_t head_size,
    float output_scale, float *output)
{
    const __m256 decay_vector = _mm256_set1_ps(decay);
    int64_t row;

    for (row = 0; row < head_size; ++row) {
        float *state_row = state + row * head_size;
        __m256 key_dot = _mm256_setzero_ps();
        int64_t column;

#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (column = 0; column < head_size; column += 8) {
            const __m256 state_values = _mm256_loadu_ps(state_row + column);
            const __m256 key_values = _mm256_loadu_ps(key + column);
            const __m256 decayed_state = _mm256_mul_ps(state_values, decay_vector);
            _mm256_storeu_ps(state_row + column,
                decayed_state);
            key_dot = _mm256_fmadd_ps(decayed_state, key_values, key_dot);
        }
        {
            const float delta = (value[row] - horizontal_sum(key_dot)) * beta;
            const __m256 delta_vector = _mm256_set1_ps(delta);
            __m256 query_dot = _mm256_setzero_ps();

#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
            for (column = 0; column < head_size; column += 8) {
                __m256 state_values = _mm256_loadu_ps(state_row + column);
                const __m256 key_values = _mm256_loadu_ps(key + column);
                const __m256 query_values = _mm256_loadu_ps(query + column);
                state_values = _mm256_fmadd_ps(delta_vector, key_values, state_values);
                _mm256_storeu_ps(state_row + column, state_values);
                query_dot = _mm256_fmadd_ps(state_values, query_values, query_dot);
            }
            output[row] = horizontal_sum(query_dot) * output_scale;
        }
    }
}

static void gdn_step_scalar(float *state, const float *key, const float *value,
    const float *query, float decay, float beta, int64_t head_size,
    float output_scale, float *output)
{
    int64_t row;

    for (row = 0; row < head_size; ++row) {
        float dot = 0.0f;
        int64_t column;
        for (column = 0; column < head_size; ++column) {
            float *entry = state + row * head_size + column;
            *entry *= decay;
            dot += *entry * key[column];
        }
        {
            const float delta = (value[row] - dot) * beta;
            float query_dot = 0.0f;
            for (column = 0; column < head_size; ++column) {
                float *entry = state + row * head_size + column;
                *entry += delta * key[column];
                query_dot += *entry * query[column];
            }
            output[row] = query_dot * output_scale;
        }
    }
}

void fortai_gdn_step(float *state, const float *key, const float *value,
    const float *query, float decay, float beta, int64_t head_size,
    float output_scale, float *output)
{
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx2") && head_size >= 8 && head_size % 8 == 0) {
        gdn_step_avx2(state, key, value, query, decay, beta, head_size,
            output_scale, output);
        return;
    }
#endif
    gdn_step_scalar(state, key, value, query, decay, beta, head_size,
        output_scale, output);
}

__attribute__((target("avx2,fma")))
static void silu_array_avx2(float *values, int64_t count)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 lower = _mm256_set1_ps(-20.0f);
    const __m256 upper = _mm256_set1_ps(20.0f);
    int64_t index;

    for (index = 0; index + 8 <= count; index += 8) {
        const __m256 input = _mm256_loadu_ps(values + index);
        const __m256 exponent = expf_avx2(_mm256_sub_ps(
            _mm256_setzero_ps(), input));
        __m256 result = _mm256_div_ps(input, _mm256_add_ps(one, exponent));
        result = _mm256_blendv_ps(result, _mm256_setzero_ps(),
            _mm256_cmp_ps(input, lower, _CMP_LT_OQ));
        result = _mm256_blendv_ps(result, input,
            _mm256_cmp_ps(input, upper, _CMP_GT_OQ));
        _mm256_storeu_ps(values + index, result);
    }
    for (; index < count; ++index)
        values[index] = values[index] / (1.0f + expf(-values[index]));
}

static void silu_array_scalar(float *values, int64_t count)
{
    int64_t index;

    for (index = 0; index < count; ++index)
        values[index] = values[index] / (1.0f + expf(-values[index]));
}

void fortai_silu(float *values, int64_t count)
{
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx2") && count >= 8) {
        silu_array_avx2(values, count);
        return;
    }
#endif
    silu_array_scalar(values, count);
}

__attribute__((target("avx2,fma")))
static void silu_product_avx2(float *left, const float *right, int64_t count)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 lower = _mm256_set1_ps(-20.0f);
    const __m256 upper = _mm256_set1_ps(20.0f);
    int64_t index;

    for (index = 0; index + 8 <= count; index += 8) {
        const __m256 input = _mm256_loadu_ps(left + index);
        const __m256 exponent = expf_avx2(_mm256_sub_ps(
            _mm256_setzero_ps(), input));
        __m256 result = _mm256_div_ps(input, _mm256_add_ps(one, exponent));
        result = _mm256_blendv_ps(result, _mm256_setzero_ps(),
            _mm256_cmp_ps(input, lower, _CMP_LT_OQ));
        result = _mm256_blendv_ps(result, input,
            _mm256_cmp_ps(input, upper, _CMP_GT_OQ));
        result = _mm256_mul_ps(result, _mm256_loadu_ps(right + index));
        _mm256_storeu_ps(left + index, result);
    }
    for (; index < count; ++index)
        left[index] = left[index] / (1.0f + expf(-left[index])) * right[index];
}

static void silu_product_scalar(float *left, const float *right, int64_t count)
{
    int64_t index;

    for (index = 0; index < count; ++index)
        left[index] = left[index] / (1.0f + expf(-left[index])) * right[index];
}

void fortai_silu_product(float *left, const float *right, int64_t count)
{
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx2") && count >= 8) {
        silu_product_avx2(left, right, count);
        return;
    }
#endif
    silu_product_scalar(left, right, count);
}
