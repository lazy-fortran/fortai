#ifndef FORTAI_CUDA_Q4_BACKEND_H
#define FORTAI_CUDA_Q4_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include "fortai_cuda_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fortai_cuda_q4_context fortai_cuda_q4_context;
typedef struct fortai_cuda_q4_weights fortai_cuda_q4_weights;

int fortai_cuda_q4_context_create(int first_device, int second_device,
    fortai_cuda_q4_context **context);
int fortai_cuda_q4_context_destroy(fortai_cuda_q4_context *context);
int fortai_cuda_q4_context_synchronize(fortai_cuda_q4_context *context);
/* Attach the downstream CUDA stream that consumes resident Q4 results.  The
 * stream handle is borrowed; passing a null handle restores the conservative
 * device-wide synchronization fallback. */
int fortai_cuda_q4_context_set_consumer_stream(fortai_cuda_q4_context *context,
    int device_slot, void *stream);
int fortai_cuda_q4_weights_upload(fortai_cuda_q4_context *context, int value_type,
    const void *host_weights, size_t weight_bytes, int rows, int width, int device,
    fortai_cuda_q4_weights **weights);
int fortai_cuda_q4_weights_destroy(fortai_cuda_q4_weights *weights);
int fortai_cuda_q4_matvec_host(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t output_bytes,
    float *elapsed_ms);

/* Queue all transfers and projections that consume one activation before
 * synchronizing each participating device. */
int fortai_cuda_q4_matvec_host_pair(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *first_weights,
    const fortai_cuda_q4_weights *second_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *elapsed_ms);

int fortai_cuda_q4_matvec_host_triplet(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *first_weights,
    const fortai_cuda_q4_weights *second_weights,
    const fortai_cuda_q4_weights *third_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *third_output,
    size_t third_output_bytes, float *elapsed_ms);

/* Execute a quantized matrix-vector product without crossing the host
 * boundary.  The activation and output pointers must be device pointers on
 * the same CUDA device as the uploaded weights.  The GGML CUDA kernel remains
 * the quantization-specific implementation; FortAI owns the surrounding
 * device pipeline. */
int fortai_cuda_q4_matvec_device(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements);

/* Execute up to three projections that consume the same resident activation.
 * The implementation builds one cached GGML graph per participating device;
 * remote weights use one peer activation copy for the whole group. */
int fortai_cuda_q4_matvec_device_group(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights * const *weights, const void *device_activation,
    size_t activation_elements, void * const *device_outputs,
    const size_t *output_elements, int count);

/* The split-attention path keeps a projection group on the second CUDA
 * device.  These variants avoid copying its Q/K/V results back to the
 * primary device merely to copy them into the attention state again. */
int fortai_cuda_q4_matvec_device_group_remote_output(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights * const *weights, const void *device_activation,
    size_t activation_elements, void * const *device_outputs,
    const size_t *output_elements, int count);
int fortai_cuda_q4_matvec_device_remote_input(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements);

int fortai_cuda_q4_embedding_device(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, int64_t token_id,
    void *device_output, size_t output_elements);

#ifdef __cplusplus
}
#endif

#endif
