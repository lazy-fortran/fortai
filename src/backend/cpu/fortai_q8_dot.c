#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#if defined(__GNUC__)
#pragma GCC optimize("O3")
#endif

extern __m256 fortai_expf_avx2(__m256 value) __asm__("_ZGVdN8v_expf")
    __attribute__((weak));

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

static float q8_dot_scalar(const int8_t *weights, const int8_t *quantized,
    const float *scales, int64_t row, int64_t block_count)
{
    float result = 0.0f;
    int64_t block;

    for (block = 0; block < block_count; ++block) {
        const int64_t offset = (row * block_count + block) * 34;
        const uint16_t scale_bits = (uint16_t)(uint8_t)weights[offset] |
            ((uint16_t)(uint8_t)weights[offset + 1] << 8);
        int32_t dot = 0;
        int i;

        for (i = 0; i < 32; ++i)
            dot += (int32_t)weights[offset + 2 + i] *
                (int32_t)quantized[block * 32 + i];
        result += half_to_float(scale_bits) * scales[block] * (float)dot;
    }
    return result;
}

__attribute__((target("avx2,f16c,fma")))
static float q8_dot_avx2(const int8_t *weights, const int8_t *quantized,
    const float *scales, int64_t row, int64_t block_count)
{
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 accumulator = _mm256_setzero_ps();
    const int8_t *row_weights = weights + row * block_count * 34 + 2;
    const int8_t *activation_values = quantized;
    const float *activation_scales = scales;
    int64_t block;

    for (block = 0; block < block_count; ++block) {
        const uint16_t scale_bits = (uint16_t)(uint8_t)row_weights[-2] |
            ((uint16_t)(uint8_t)row_weights[-1] << 8);
        const __m256i weight = _mm256_loadu_si256(
            (const __m256i *)row_weights);
        const __m256i activation = _mm256_loadu_si256(
            (const __m256i *)activation_values);
        const __m256i absolute_weight = _mm256_abs_epi8(weight);
        const __m256i signed_activation = _mm256_sign_epi8(activation, weight);
        const __m256i products = _mm256_maddubs_epi16(absolute_weight,
            signed_activation);
        const __m256i pairs = _mm256_madd_epi16(products, ones);
        const __m256 dot = _mm256_cvtepi32_ps(pairs);
        const __m256 scale = _mm256_set1_ps(_cvtsh_ss(scale_bits) * activation_scales[0]);
        accumulator = _mm256_fmadd_ps(scale, dot, accumulator);
        row_weights += 34;
        activation_values += 32;
        ++activation_scales;
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

float fortai_q8_dot(const int8_t *weights, const int8_t *quantized,
    const float *scales, int64_t row, int64_t block_count)
{
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx2"))
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
        const __m256 exponent = fortai_expf_avx2(_mm256_sub_ps(
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
    if (__builtin_cpu_supports("avx2") && fortai_expf_avx2 != 0 && count >= 8) {
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
        const __m256 exponent = fortai_expf_avx2(_mm256_sub_ps(
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
    if (__builtin_cpu_supports("avx2") && fortai_expf_avx2 != 0 && count >= 8) {
        silu_product_avx2(left, right, count);
        return;
    }
#endif
    silu_product_scalar(left, right, count);
}
