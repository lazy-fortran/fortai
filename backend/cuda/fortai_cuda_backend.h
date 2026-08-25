#ifndef FORTAI_CUDA_BACKEND_H
#define FORTAI_CUDA_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fortai_cuda_q8_context fortai_cuda_q8_context;
typedef struct fortai_cuda_q8_weights fortai_cuda_q8_weights;
typedef struct fortai_cuda_qwen35_recurrent fortai_cuda_qwen35_recurrent;
typedef struct fortai_cuda_qwen35_attention fortai_cuda_qwen35_attention;

enum {
    FORTAI_CUDA_OK = 0,
    FORTAI_CUDA_INVALID = 1,
    FORTAI_CUDA_RUNTIME_ERROR = 2
};

int fortai_cuda_q8_context_create(int device, fortai_cuda_q8_context **context);
int fortai_cuda_q8_context_destroy(fortai_cuda_q8_context *context);
int fortai_cuda_memory_info(int device, size_t *free_bytes, size_t *total_bytes);
int fortai_cuda_q8_context_set_position(fortai_cuda_q8_context *context, int position);
int fortai_cuda_q8_context_synchronize(fortai_cuda_q8_context *context);
/* Return the native CUDA stream used by this context for explicit stream
 * hand-off with another CUDA backend.  The returned handle is borrowed. */
void *fortai_cuda_q8_context_stream(fortai_cuda_q8_context *context);
int fortai_cuda_q8_context_capture_begin(fortai_cuda_q8_context *context);
int fortai_cuda_q8_context_capture_end(fortai_cuda_q8_context *context);
int fortai_cuda_q8_context_graph_launch(fortai_cuda_q8_context *context);

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
int fortai_cuda_q8_device_buffer_upload_ptr(fortai_cuda_q8_context *context,
    void *device_buffer, const void *host_data, size_t bytes);
int fortai_cuda_q8_device_buffer_download(fortai_cuda_q8_context *context,
    void *host_data, const void *device_buffer, size_t bytes);

/* Resident operation: weights, activation, and output are all device
 * pointers. kernel_ms measures the device kernel only. */
int fortai_cuda_q8_matvec_resident(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *device_activation,
    void *device_output, float *kernel_ms);
int fortai_cuda_q8_matvec_device_f32(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements);
int fortai_cuda_qwen35_embedding_device(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, int64_t token_id, void *device_output,
    size_t output_elements);

int fortai_cuda_qwen35_copy_device(fortai_cuda_q8_context *context,
    const void *device_input, void *device_output, size_t bytes);
int fortai_cuda_qwen35_add_device(fortai_cuda_q8_context *context,
    const void *device_left, const void *device_right, void *device_output,
    size_t elements);
int fortai_cuda_qwen35_rms_norm_device(fortai_cuda_q8_context *context,
    const void *device_input, const void *device_weights, void *device_output,
    size_t elements, float epsilon);
int fortai_cuda_qwen35_silu_product_device(fortai_cuda_q8_context *context,
    void *device_gate, const void *device_up, size_t elements);
int fortai_cuda_qwen35_argmax_device(fortai_cuda_q8_context *context,
    const void *device_logits, size_t elements, int *host_index);

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

int fortai_cuda_q8_matvec_host_triplet_contiguous(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *first_weights,
    const fortai_cuda_q8_weights *second_weights,
    const fortai_cuda_q8_weights *third_weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t host_output_bytes,
    float *elapsed_ms);

int fortai_cuda_q8_ffn_host(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *gate_weights,
    const fortai_cuda_q8_weights *up_weights,
    const fortai_cuda_q8_weights *down_weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t output_bytes,
    float *elapsed_ms);

int fortai_cuda_q8_ffn_device(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *gate_weights,
    const fortai_cuda_q8_weights *up_weights,
    const fortai_cuda_q8_weights *down_weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements);

/* A Qwen3.5 recurrent layer keeps its convolution and GDN state on-device.
 * The run call accepts one host Q8 activation and returns only the layer
 * output after the recurrent projection; all intermediate projections,
 * convolution, GDN update, and ssm output GEMV stay on the CUDA stream. */
int fortai_cuda_qwen35_recurrent_create(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *qkv_weights,
    const fortai_cuda_q8_weights *gate_weights,
    const fortai_cuda_q8_weights *alpha_weights,
    const fortai_cuda_q8_weights *beta_weights,
    const fortai_cuda_q8_weights *output_weights,
    const void *conv_weights, size_t conv_weight_bytes, int conv_size, int conv_kernel,
    const void *ssm_a, size_t ssm_a_bytes, const void *ssm_dt, size_t ssm_dt_bytes,
    const void *ssm_norm, size_t ssm_norm_bytes, int state_size, int key_heads,
    int value_heads, int head_size, int inner_size, float norm_epsilon,
    fortai_cuda_qwen35_recurrent **layer);
int fortai_cuda_qwen35_recurrent_create_state(fortai_cuda_q8_context *context,
    const void *conv_weights, size_t conv_weight_bytes, int conv_size, int conv_kernel,
    const void *ssm_a, size_t ssm_a_bytes, const void *ssm_dt, size_t ssm_dt_bytes,
    const void *ssm_norm, size_t ssm_norm_bytes, int state_size, int key_heads,
    int value_heads, int head_size, int inner_size, float norm_epsilon,
    fortai_cuda_qwen35_recurrent **layer);
int fortai_cuda_qwen35_recurrent_destroy(fortai_cuda_qwen35_recurrent *layer);
int fortai_cuda_qwen35_recurrent_reset(fortai_cuda_qwen35_recurrent *layer);
int fortai_cuda_qwen35_recurrent_run(fortai_cuda_qwen35_recurrent *layer,
    const void *host_activation, size_t activation_bytes, float *host_output,
    size_t output_bytes, float *elapsed_ms);
int fortai_cuda_qwen35_recurrent_run_device(fortai_cuda_qwen35_recurrent *layer,
    const void *device_activation, size_t activation_elements,
    void *device_output, size_t output_elements);
int fortai_cuda_qwen35_recurrent_run_core_device(fortai_cuda_qwen35_recurrent *layer,
    void *device_qkv, size_t qkv_elements, const void *device_gate,
    size_t gate_elements, const void *device_alpha, size_t alpha_elements,
    const void *device_beta, size_t beta_elements, void *device_output,
    size_t output_elements);

int fortai_cuda_qwen35_attention_create(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *query_weights,
    const fortai_cuda_q8_weights *key_weights,
    const fortai_cuda_q8_weights *value_weights,
    const fortai_cuda_q8_weights *output_weights,
    const void *query_norm, size_t query_norm_bytes,
    const void *key_norm, size_t key_norm_bytes,
    int heads, int key_value_heads, int head_size, int value_size,
    int max_context, int rope_dimension, float rope_base, float norm_epsilon,
    fortai_cuda_qwen35_attention **layer);
int fortai_cuda_qwen35_attention_create_state(fortai_cuda_q8_context *context,
    const void *query_norm, size_t query_norm_bytes, const void *key_norm,
    size_t key_norm_bytes, int heads, int key_value_heads, int head_size,
    int value_size, int max_context, int rope_dimension, float rope_base,
    float norm_epsilon, fortai_cuda_qwen35_attention **layer);
int fortai_cuda_qwen35_attention_destroy(fortai_cuda_qwen35_attention *layer);
int fortai_cuda_qwen35_attention_reset(fortai_cuda_qwen35_attention *layer);
int fortai_cuda_qwen35_attention_run_device(fortai_cuda_qwen35_attention *layer,
    const void *device_activation, size_t activation_elements, int position,
    void *device_output, size_t output_elements);
int fortai_cuda_qwen35_attention_run_core_device(fortai_cuda_qwen35_attention *layer,
    const void *device_query, size_t query_elements, const void *device_key,
    size_t key_elements, const void *device_value, size_t value_elements,
    int position, void *device_output, size_t output_elements);

const char *fortai_cuda_q8_last_error(const fortai_cuda_q8_context *context);

#ifdef __cplusplus
}
#endif

#endif
