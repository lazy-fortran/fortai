#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#if defined(__GNUC__)
#pragma GCC optimize("O3")
#endif

__attribute__((target("avx2,fma")))
static inline __m256 expf_avx2(__m256 value)
{
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(value,
        _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(n,
        _mm256_set1_ps(0x1.7f7d1cp-20f),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), value));
    const __m256i exponent = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 scale = _mm256_castsi256_ps(_mm256_add_epi32(exponent,
        _mm256_castps_si256(_mm256_set1_ps(1.0f))));
    const __m256i overflow = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
        _mm256_set1_ps(126.0f), _CMP_GT_OQ));
    const __m256 square = _mm256_mul_ps(b, b);
    const __m256 polynomial = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                _mm256_set1_ps(0x1.573e2ep-5f)), square,
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                _mm256_set1_ps(0x1.fffdb6p-2f))),
        square, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));

    if (!_mm256_movemask_ps(_mm256_castsi256_ps(overflow)))
        return _mm256_fmadd_ps(polynomial, scale, scale);

    {
        const __m256i underflow_adjustment = _mm256_and_si256(
            _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(),
                _CMP_LE_OQ)), _mm256_set1_epi32((int32_t)0x82000000u));
        const __m256 first_scale = _mm256_castsi256_ps(_mm256_add_epi32(
            underflow_adjustment, _mm256_set1_epi32(0x7f000000)));
        const __m256 second_scale = _mm256_castsi256_ps(_mm256_sub_epi32(
            exponent, underflow_adjustment));
        const __m256i extreme = _mm256_castps_si256(_mm256_cmp_ps(
            _mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
            _mm256_set1_ps(192.0f), _CMP_GT_OQ));
        return _mm256_or_ps(
            _mm256_and_ps(_mm256_castsi256_ps(extreme),
                _mm256_mul_ps(first_scale, first_scale)),
            _mm256_andnot_ps(_mm256_castsi256_ps(extreme),
                _mm256_or_ps(
                    _mm256_and_ps(_mm256_castsi256_ps(overflow),
                        _mm256_mul_ps(_mm256_fmadd_ps(second_scale,
                            polynomial, second_scale), first_scale)),
                    _mm256_andnot_ps(_mm256_castsi256_ps(overflow),
                        _mm256_fmadd_ps(scale, polynomial, scale)))));
    }
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
    const uint32_t exponent_bits = (bits >> 23) & 0xffu;
    uint32_t fraction = bits & 0x007fffffu;
    int32_t exponent;

    if (exponent_bits == 0xffu) {
        if (fraction == 0u)
            return (uint16_t)(sign | 0x7c00u);
        fraction >>= 13;
        if (fraction == 0u)
            fraction = 1u;
        return (uint16_t)(sign | 0x7c00u | fraction);
    }
    if (exponent_bits == 0u)
        return (uint16_t)sign;

    exponent = (int32_t)exponent_bits - 127;
    if (exponent < -14) {
        uint32_t halfway, remainder, rounded;
        int32_t shift;
        if (exponent < -25)
            return (uint16_t)sign;
        fraction |= 0x00800000u;
        shift = -exponent - 1;
        rounded = fraction >> shift;
        remainder = fraction & ((1u << shift) - 1u);
        halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0u))
            ++rounded;
        return (uint16_t)(sign | rounded);
    }
    if (exponent > 15)
        return (uint16_t)(sign | 0x7c00u);

    {
        uint32_t rounded = fraction >> 13;
        const uint32_t remainder = fraction & 0x1fffu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u) != 0u))
            ++rounded;
        if (rounded == 0x400u) {
            rounded = 0u;
            ++exponent;
            if (exponent > 15)
                return (uint16_t)(sign | 0x7c00u);
        }
        return (uint16_t)(sign | ((uint32_t)(exponent + 15) << 10) | rounded);
    }
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

static void floats_to_half_scalar(const float *__restrict input,
    uint16_t *__restrict output, int64_t count)
{
    int64_t i;
    for (i = 0; i < count; ++i)
        output[i] = float_to_half(input[i]);
}

__attribute__((target("avx2,f16c")))
static void floats_to_half_avx2(const float *__restrict input,
    uint16_t *__restrict output, int64_t count)
{
    int64_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 values = _mm256_loadu_ps(input + i);
        _mm_storeu_si128((__m128i *)(output + i), _mm256_cvtps_ph(values, 0));
    }
    for (; i < count; ++i)
        output[i] = _cvtss_sh(input[i], _MM_FROUND_TO_NEAREST_INT);
}

static float dot_f16_scalar(const uint16_t *__restrict left,
    const uint16_t *__restrict right, int64_t count)
{
    double sum = 0.0;
    int64_t i;
    for (i = 0; i < count; ++i)
        sum += (double)(half_to_float(left[i]) * half_to_float(right[i]));
    return (float)sum;
}

__attribute__((target("avx2,f16c,fma")))
static float dot_f16_avx2(const uint16_t *__restrict left,
    const uint16_t *__restrict right, int64_t count)
{
    __m256 sums[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()
    };
    int64_t i = 0;
    int j;
    double sum;

    for (; i + 32 <= count; i += 32) {
        for (j = 0; j < 4; ++j) {
            const __m256 left_values = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(left + i + 8 * j)));
            const __m256 right_values = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(right + i + 8 * j)));
            sums[j] = _mm256_fmadd_ps(left_values, right_values, sums[j]);
        }
    }

    sums[0] = _mm256_add_ps(sums[0], sums[2]);
    sums[1] = _mm256_add_ps(sums[1], sums[3]);
    sums[0] = _mm256_add_ps(sums[0], sums[1]);
    {
        const __m128 lanes = _mm_add_ps(_mm256_castps256_ps128(sums[0]),
            _mm256_extractf128_ps(sums[0], 1));
        const __m128 pairs = _mm_hadd_ps(lanes, lanes);
        sum = (double)_mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
    }
    for (; i < count; ++i)
        sum += (double)(half_to_float(left[i]) * half_to_float(right[i]));
    return (float)sum;
}

static void scale_f16_scalar(uint16_t *__restrict values, int64_t count,
    float factor)
{
    int64_t i;
    for (i = 0; i < count; ++i)
        values[i] = float_to_half(half_to_float(values[i]) * factor);
}

__attribute__((target("avx2,f16c")))
static void scale_f16_avx2(uint16_t *__restrict values, int64_t count,
    float factor)
{
    const __m256 factor_values = _mm256_set1_ps(factor);
    int64_t i = 0;
    int j;
    for (; i + 32 <= count; i += 32) {
        for (j = 0; j < 4; ++j) {
            const __m256 values_f32 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(values + i + 8 * j)));
            _mm_storeu_si128((__m128i *)(values + i + 8 * j),
                _mm256_cvtps_ph(_mm256_mul_ps(values_f32, factor_values), 0));
        }
    }
    for (; i < count; ++i)
        values[i] = _cvtss_sh(_cvtsh_ss(values[i]) * factor,
            _MM_FROUND_TO_NEAREST_INT);
}

static void mad_f16_scalar(uint16_t *__restrict output,
    const uint16_t *__restrict input, int64_t count, float factor)
{
    int64_t i;
    for (i = 0; i < count; ++i)
        output[i] = float_to_half(half_to_float(output[i]) +
            half_to_float(input[i]) * factor);
}

__attribute__((target("avx2,f16c,fma")))
static void mad_f16_avx2(uint16_t *__restrict output,
    const uint16_t *__restrict input, int64_t count, float factor)
{
    const __m256 factor_values = _mm256_set1_ps(factor);
    int64_t i = 0;
    int j;
    for (; i + 32 <= count; i += 32) {
        for (j = 0; j < 4; ++j) {
            const __m256 input_f32 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(input + i + 8 * j)));
            const __m256 output_f32 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *)(output + i + 8 * j)));
            _mm_storeu_si128((__m128i *)(output + i + 8 * j),
                _mm256_cvtps_ph(_mm256_fmadd_ps(input_f32, factor_values,
                    output_f32), 0));
        }
    }
    for (; i < count; ++i)
        output[i] = _cvtss_sh(_cvtsh_ss(output[i]) +
            _cvtsh_ss(input[i]) * factor, _MM_FROUND_TO_NEAREST_INT);
}

void fortai_flash_attention_f16(const float *__restrict query,
    const float *__restrict key_cache, const float *__restrict value_cache,
    int64_t count, int64_t key_stride, int64_t value_stride,
    int64_t key_size, int64_t value_size, float scale,
    float *__restrict output)
{
    typedef void (*convert_fn)(const float *, uint16_t *, int64_t);
    typedef float (*dot_fn)(const uint16_t *, const uint16_t *, int64_t);
    typedef void (*scale_fn)(uint16_t *, int64_t, float);
    typedef void (*mad_fn)(uint16_t *, const uint16_t *, int64_t, float);
    convert_fn convert = floats_to_half_scalar;
    dot_fn dot = dot_f16_scalar;
    scale_fn scale_values = scale_f16_scalar;
    mad_fn add_scaled = mad_f16_scalar;
    float sum = 0.0f;
    float maximum = -INFINITY;
    int64_t cache_index, i;

    if (count <= 0 || key_stride < key_size || value_stride < value_size ||
            key_size <= 0 || value_size <= 0)
        return;

#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") &&
            __builtin_cpu_supports("fma")) {
        convert = floats_to_half_avx2;
        dot = dot_f16_avx2;
        scale_values = scale_f16_avx2;
        add_scaled = mad_f16_avx2;
    }
#endif

    {
        uint16_t query_f16[key_size];
        uint16_t key_f16[key_size];
        uint16_t value_f16[value_size];
        uint16_t accumulator_f16[value_size];

        convert(query, query_f16, key_size);
        for (i = 0; i < value_size; ++i)
            accumulator_f16[i] = 0;

        for (cache_index = 0; cache_index < count; ++cache_index) {
            const float old_maximum = maximum;
            float maximum_scale = 1.0f;
            float value_scale = 1.0f;
            float score;

            convert(key_cache + cache_index * key_stride, key_f16, key_size);
            score = dot(key_f16, query_f16, key_size) * scale;
            if (score > maximum) {
                maximum = score;
                maximum_scale = expf(old_maximum - maximum);
                scale_values(accumulator_f16, value_size, maximum_scale);
            } else {
                value_scale = expf(score - maximum);
            }

            convert(value_cache + cache_index * value_stride, value_f16, value_size);
            add_scaled(accumulator_f16, value_f16, value_size, value_scale);
            sum = fmaf(sum, maximum_scale, value_scale);
        }

        {
            const float inverse_sum = sum == 0.0f ? 0.0f : 1.0f / sum;
            for (i = 0; i < value_size; ++i)
                output[i] = half_to_float(accumulator_f16[i]) * inverse_sum;
        }
    }
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
        int32_t dot = 0;
        int i;

        for (i = 0; i < 32; ++i)
            dot += (int32_t)weights[offset + 2 + i] *
                (int32_t)quantized[activation_offset + 2 + i];
        result += half_to_float(scale_bits) * scales[block] *
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

    #if defined(__GNUC__)
    #pragma GCC unroll 2
    #endif
    for (block = 0; block < blocks; ++block) {
        const uint16_t scale_bits = load_u16(row_start + offset);
        const __m256i weight = _mm256_loadu_si256(
            (const __m256i *)(row_start + offset + 2));
        const __m256i activation = _mm256_loadu_si256(
            (const __m256i *)(activation_start + offset + 2));
        /* The quantizer has already converted this activation scale to the
         * exact FP32 value in scales[]. Reusing it avoids a second FP16
         * conversion for every weight block in every matvec. */
        const __m256 scale = _mm256_set1_ps(_cvtsh_ss(scale_bits) * scales[block]);
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
static inline float reduce_dot_sums(__m256 sums[4])
{
    __m128 lanes, pairs;

    sums[0] = _mm256_add_ps(sums[0], sums[2]);
    sums[1] = _mm256_add_ps(sums[1], sums[3]);
    sums[0] = _mm256_add_ps(sums[0], sums[1]);
    lanes = _mm_add_ps(_mm256_castps256_ps128(sums[0]),
        _mm256_extractf128_ps(sums[0], 1));
    pairs = _mm_hadd_ps(lanes, lanes);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
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
        __m256 key_sums[4] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps(), _mm256_setzero_ps()
        };
        int64_t column = 0;
        int j;

#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
        for (; column + 32 <= head_size; column += 32) {
            for (j = 0; j < 4; ++j) {
                const __m256 state_values = _mm256_loadu_ps(state_row + column + 8 * j);
                const __m256 key_values = _mm256_loadu_ps(key + column + 8 * j);
                const __m256 decayed_state = _mm256_mul_ps(state_values, decay_vector);
                _mm256_storeu_ps(state_row + column + 8 * j, decayed_state);
                key_sums[j] = _mm256_fmadd_ps(decayed_state, key_values, key_sums[j]);
            }
        }
        {
            float key_dot = reduce_dot_sums(key_sums);
            for (; column < head_size; ++column) {
                state_row[column] *= decay;
                key_dot += state_row[column] * key[column];
            }
            const float delta = (value[row] - key_dot) * beta;
            const __m256 delta_vector = _mm256_set1_ps(delta);
            __m256 query_sums[4] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps()
            };

            column = 0;
#if defined(__GNUC__)
#pragma GCC unroll 4
#endif
            for (; column + 32 <= head_size; column += 32) {
                for (j = 0; j < 4; ++j) {
                    __m256 state_values = _mm256_loadu_ps(state_row + column + 8 * j);
                    const __m256 key_values = _mm256_loadu_ps(key + column + 8 * j);
                    const __m256 query_values = _mm256_loadu_ps(query + column + 8 * j);
                    state_values = _mm256_fmadd_ps(delta_vector, key_values, state_values);
                    _mm256_storeu_ps(state_row + column + 8 * j, state_values);
                    query_sums[j] = _mm256_fmadd_ps(state_values, query_values, query_sums[j]);
                }
            }
            {
                float query_dot = reduce_dot_sums(query_sums);
                for (; column < head_size; ++column) {
                    state_row[column] += delta * key[column];
                    query_dot += state_row[column] * query[column];
                }
                output[row] = query_dot * output_scale;
            }
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
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
            head_size >= 8 && head_size % 8 == 0) {
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
    int64_t index;

    for (index = 0; index + 8 <= count; index += 8) {
        const __m256 input = _mm256_loadu_ps(values + index);
        const __m256 exponent = expf_avx2(_mm256_sub_ps(
            _mm256_setzero_ps(), input));
        __m256 result = _mm256_div_ps(input, _mm256_add_ps(one, exponent));
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
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && count >= 8) {
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
    int64_t index;

    for (index = 0; index + 8 <= count; index += 8) {
        const __m256 input = _mm256_loadu_ps(left + index);
        const __m256 exponent = expf_avx2(_mm256_sub_ps(
            _mm256_setzero_ps(), input));
        __m256 result = _mm256_div_ps(input, _mm256_add_ps(one, exponent));
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
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && count >= 8) {
        silu_product_avx2(left, right, count);
        return;
    }
#endif
    silu_product_scalar(left, right, count);
}
