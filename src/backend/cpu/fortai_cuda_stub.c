#include "../../../backend/cuda/fortai_cuda_backend.h"

#if defined(__GNUC__)
#define FORTAI_WEAK __attribute__((weak))
#else
#define FORTAI_WEAK
#endif

static int unavailable(void) {
    return FORTAI_CUDA_RUNTIME_ERROR;
}

int FORTAI_WEAK fortai_cuda_q8_context_create(int device,
    fortai_cuda_q8_context **context) {
    (void)device;
    if (context) *context = NULL;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_context_destroy(fortai_cuda_q8_context *context) {
    (void)context;
    return FORTAI_CUDA_OK;
}

int FORTAI_WEAK fortai_cuda_q8_weights_upload(fortai_cuda_q8_context *context,
    const void *host_weights, size_t weight_bytes, int rows, int width,
    fortai_cuda_q8_weights **weights) {
    (void)context;
    (void)host_weights;
    (void)weight_bytes;
    (void)rows;
    (void)width;
    if (weights) *weights = NULL;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_weights_destroy(fortai_cuda_q8_weights *weights) {
    (void)weights;
    return FORTAI_CUDA_OK;
}

int FORTAI_WEAK fortai_cuda_q8_device_buffer_create(fortai_cuda_q8_context *context,
    size_t bytes, void **device_buffer) {
    (void)context;
    (void)bytes;
    if (device_buffer) *device_buffer = NULL;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_device_buffer_destroy(fortai_cuda_q8_context *context,
    void *device_buffer) {
    (void)context;
    (void)device_buffer;
    return FORTAI_CUDA_OK;
}

int FORTAI_WEAK fortai_cuda_q8_device_buffer_upload(fortai_cuda_q8_context *context,
    void *device_buffer, const void *host_data, size_t bytes) {
    (void)context;
    (void)device_buffer;
    (void)host_data;
    (void)bytes;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_device_buffer_download(fortai_cuda_q8_context *context,
    void *host_data, const void *device_buffer, size_t bytes) {
    (void)context;
    (void)host_data;
    (void)device_buffer;
    (void)bytes;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_matvec_resident(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *device_activation,
    void *device_output, float *kernel_ms) {
    (void)context;
    (void)weights;
    (void)device_activation;
    (void)device_output;
    if (kernel_ms) *kernel_ms = 0.0f;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_matvec_host(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t output_bytes,
    float *elapsed_ms) {
    (void)context;
    (void)weights;
    (void)host_activation;
    (void)activation_bytes;
    (void)host_output;
    (void)output_bytes;
    if (elapsed_ms) *elapsed_ms = 0.0f;
    return unavailable();
}

const char * FORTAI_WEAK fortai_cuda_q8_last_error(
    const fortai_cuda_q8_context *context) {
    (void)context;
    return "CUDA backend is not linked";
}
