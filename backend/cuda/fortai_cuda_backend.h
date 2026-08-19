#ifndef FORTAI_CUDA_BACKEND_H
#define FORTAI_CUDA_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fortai_cuda_q8_context fortai_cuda_q8_context;
typedef struct fortai_cuda_q8_weights fortai_cuda_q8_weights;

enum {
    FORTAI_CUDA_OK = 0,
    FORTAI_CUDA_INVALID = 1,
    FORTAI_CUDA_RUNTIME_ERROR = 2
};

int fortai_cuda_q8_context_create(int device, fortai_cuda_q8_context **context);
int fortai_cuda_q8_context_destroy(fortai_cuda_q8_context *context);

int fortai_cuda_q8_weights_upload(fortai_cuda_q8_context *context,
    const void *host_weights, size_t weight_bytes, int rows, int width,
    fortai_cuda_q8_weights **weights);
int fortai_cuda_q8_weights_destroy(fortai_cuda_q8_weights *weights);

/* Device buffers are opaque to the Fortran side and remain resident until
 * fortai_cuda_q8_device_buffer_destroy is called. */
int fortai_cuda_q8_device_buffer_create(fortai_cuda_q8_context *context,
    size_t bytes, void **device_buffer);
int fortai_cuda_q8_device_buffer_destroy(fortai_cuda_q8_context *context,
    void *device_buffer);
int fortai_cuda_q8_device_buffer_upload(fortai_cuda_q8_context *context,
    void *device_buffer, const void *host_data, size_t bytes);
int fortai_cuda_q8_device_buffer_download(fortai_cuda_q8_context *context,
    void *host_data, const void *device_buffer, size_t bytes);

/* Resident operation: weights, activation, and output are all device
 * pointers. kernel_ms measures the device kernel only. */
int fortai_cuda_q8_matvec_resident(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *device_activation,
    void *device_output, float *kernel_ms);

/* Convenience operation for the first host-controlled integration slice.
 * Weights remain resident across calls; activation and output are transferred
 * for this call and elapsed_ms includes those transfers. */
int fortai_cuda_q8_matvec_host(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t output_bytes,
    float *elapsed_ms);

/* Grouped host-controlled operations: the quantized activation is uploaded
 * once, then each resident weight matrix is evaluated on the same stream. */
int fortai_cuda_q8_matvec_host_pair(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *first_weights,
    const fortai_cuda_q8_weights *second_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *elapsed_ms);

int fortai_cuda_q8_matvec_host_triplet(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *first_weights,
    const fortai_cuda_q8_weights *second_weights,
    const fortai_cuda_q8_weights *third_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *third_output,
    size_t third_output_bytes, float *elapsed_ms);

const char *fortai_cuda_q8_last_error(const fortai_cuda_q8_context *context);

#ifdef __cplusplus
}
#endif

#endif
