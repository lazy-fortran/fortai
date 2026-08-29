#ifndef FORTAI_CUDA_FATTN_BACKEND_H
#define FORTAI_CUDA_FATTN_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Native Qwen3.8 flash-attention launcher.  The implementation instantiates
 * llama.cpp's MIT-licensed low-level F16 MMA tile for the production
 * D=256/GQA=6 shape; FortAI owns tensor layout, masking, cache lifetime, the
 * Qwen attention gate, and all scheduling around it. */
size_t fortai_cuda_qwen38_fattn_mask_bytes(int batch, int active_rows);
size_t fortai_cuda_qwen38_fattn_meta_bytes(int batch, int key_value_heads,
    int active_rows, int multiprocessor_count);
int fortai_cuda_qwen38_fattn_f16(const float *query, const void *key_f16,
    const void *value_f16, float *output, void *mask_f16, void *meta,
    size_t meta_bytes, int heads, int key_value_heads, int start_position,
    int batch, int active_rows, int query_stride, int output_stride,
    int query_head_stride, int multiprocessor_count, void *stream);

#ifdef __cplusplus
}
#endif

#endif
