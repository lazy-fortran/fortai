#ifndef FORTAI_CUDA_Q4_BACKEND_H
#define FORTAI_CUDA_Q4_BACKEND_H

#include <stddef.h>
#include "fortai_cuda_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fortai_cuda_q4_context fortai_cuda_q4_context;
typedef struct fortai_cuda_q4_weights fortai_cuda_q4_weights;

int fortai_cuda_q4_context_create(int first_device, int second_device,
    fortai_cuda_q4_context **context);
int fortai_cuda_q4_context_destroy(fortai_cuda_q4_context *context);
int fortai_cuda_q4_weights_upload(fortai_cuda_q4_context *context, int value_type,
    const void *host_weights, size_t weight_bytes, int rows, int width, int device,
    fortai_cuda_q4_weights **weights);
int fortai_cuda_q4_weights_destroy(fortai_cuda_q4_weights *weights);
int fortai_cuda_q4_matvec_host(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t output_bytes,
    float *elapsed_ms);

#ifdef __cplusplus
}
#endif

#endif
