#include "../../../backend/cuda/fortai_cuda_backend.h"
#include "../../../backend/cuda/fortai_cuda_q4_backend.h"

#if defined(__GNUC__)
#define FORTAI_WEAK __attribute__((weak))
#else
#define FORTAI_WEAK
#endif

static int unavailable(void) {
    return FORTAI_CUDA_RUNTIME_ERROR;
}

int FORTAI_WEAK fortai_cuda_q4_context_create(int first_device, int second_device,
    fortai_cuda_q4_context **context) {
    (void)first_device;
    (void)second_device;
    if (context) *context = NULL;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_context_destroy(fortai_cuda_q4_context *context) {
    (void)context;
    return FORTAI_CUDA_OK;
}

int FORTAI_WEAK fortai_cuda_q4_context_synchronize(fortai_cuda_q4_context *context) {
    (void)context;
    return FORTAI_CUDA_OK;
}

int FORTAI_WEAK fortai_cuda_q4_context_set_consumer_stream(fortai_cuda_q4_context *context,
    int device_slot, void *stream) {
    (void)context;
    (void)device_slot;
    (void)stream;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_weights_upload(fortai_cuda_q4_context *context, int value_type,
    const void *host_weights, size_t weight_bytes, int rows, int width, int device,
    fortai_cuda_q4_weights **weights) {
    (void)context;
    (void)value_type;
    (void)host_weights;
    (void)weight_bytes;
    (void)rows;
    (void)width;
    (void)device;
    if (weights) *weights = NULL;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_weights_destroy(fortai_cuda_q4_weights *weights) {
    (void)weights;
    return FORTAI_CUDA_OK;
}

int FORTAI_WEAK fortai_cuda_q4_matvec_host(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t output_bytes, float *elapsed_ms) {
    (void)context;
    (void)weights;
    (void)host_activation;
    (void)activation_bytes;
    (void)host_output;
    (void)output_bytes;
    if (elapsed_ms) *elapsed_ms = 0.0f;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_matvec_host_pair(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *first_weights,
    const fortai_cuda_q4_weights *second_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *elapsed_ms) {
    (void)context;
    (void)first_weights;
    (void)second_weights;
    (void)host_activation;
    (void)activation_bytes;
    (void)first_output;
    (void)first_output_bytes;
    (void)second_output;
    (void)second_output_bytes;
    if (elapsed_ms) *elapsed_ms = 0.0f;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_matvec_host_triplet(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *first_weights,
    const fortai_cuda_q4_weights *second_weights,
    const fortai_cuda_q4_weights *third_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *third_output,
    size_t third_output_bytes, float *elapsed_ms) {
    (void)context;
    (void)first_weights;
    (void)second_weights;
    (void)third_weights;
    (void)host_activation;
    (void)activation_bytes;
    (void)first_output;
    (void)first_output_bytes;
    (void)second_output;
    (void)second_output_bytes;
    (void)third_output;
    (void)third_output_bytes;
    if (elapsed_ms) *elapsed_ms = 0.0f;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_matvec_device(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    (void)context;
    (void)weights;
    (void)device_activation;
    (void)activation_elements;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_matvec_device_swiglu(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *gate_weights, const fortai_cuda_q4_weights *up_weights,
    const void *device_activation, size_t activation_elements, void *device_output,
    size_t output_elements) {
    (void)context;
    (void)gate_weights;
    (void)up_weights;
    (void)device_activation;
    (void)activation_elements;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_matvec_device_swiglu_remote_output(
    fortai_cuda_q4_context *context, const fortai_cuda_q4_weights *gate_weights,
    const fortai_cuda_q4_weights *up_weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    (void)context;
    (void)gate_weights;
    (void)up_weights;
    (void)device_activation;
    (void)activation_elements;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_matvec_device_swiglu_down(
    fortai_cuda_q4_context *context, const fortai_cuda_q4_weights *gate_weights,
    const fortai_cuda_q4_weights *up_weights, const fortai_cuda_q4_weights *down_weights,
    const void *device_activation, size_t activation_elements, void *device_output,
    size_t output_elements) {
    (void)context;
    (void)gate_weights;
    (void)up_weights;
    (void)down_weights;
    (void)device_activation;
    (void)activation_elements;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_matvec_device_group(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights * const *weights, const void *device_activation,
    size_t activation_elements, void * const *device_outputs,
    const size_t *output_elements, int count) {
    (void)context;
    (void)weights;
    (void)device_activation;
    (void)activation_elements;
    (void)device_outputs;
    (void)output_elements;
    (void)count;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_matvec_device_group_remote_output(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights * const *weights, const void *device_activation,
    size_t activation_elements, void * const *device_outputs,
    const size_t *output_elements, int count) {
    (void)context;
    (void)weights;
    (void)device_activation;
    (void)activation_elements;
    (void)device_outputs;
    (void)output_elements;
    (void)count;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_matvec_device_remote_input(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    (void)context;
    (void)weights;
    (void)device_activation;
    (void)activation_elements;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q4_embedding_device(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, int64_t token_id,
    void *device_output, size_t output_elements) {
    (void)context;
    (void)weights;
    (void)token_id;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_context_create(int device,
    fortai_cuda_q8_context **context) {
    (void)device;
    if (context) *context = NULL;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_memory_info(int device, size_t *free_bytes,
    size_t *total_bytes) {
    (void)device;
    if (free_bytes) *free_bytes = 0;
    if (total_bytes) *total_bytes = 0;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_context_destroy(fortai_cuda_q8_context *context) {
    (void)context;
    return FORTAI_CUDA_OK;
}

int FORTAI_WEAK fortai_cuda_q8_context_set_position(fortai_cuda_q8_context *context, int position) {
    (void)context;
    (void)position;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_context_synchronize(fortai_cuda_q8_context *context) {
    (void)context;
    return unavailable();
}

void *FORTAI_WEAK fortai_cuda_q8_context_stream(fortai_cuda_q8_context *context) {
    (void)context;
    return NULL;
}

int FORTAI_WEAK fortai_cuda_q8_context_capture_begin(fortai_cuda_q8_context *context) {
    (void)context;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_context_capture_end(fortai_cuda_q8_context *context) {
    (void)context;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_context_graph_launch(fortai_cuda_q8_context *context) {
    (void)context;
    return unavailable();
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

int FORTAI_WEAK fortai_cuda_q8_device_buffer_upload_ptr(fortai_cuda_q8_context *context,
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

int FORTAI_WEAK fortai_cuda_q8_matvec_device_f32(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    (void)context;
    (void)weights;
    (void)device_activation;
    (void)activation_elements;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_embedding_device(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *weights, int64_t token_id, void *device_output,
    size_t output_elements) {
    (void)context;
    (void)weights;
    (void)token_id;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_copy_device(fortai_cuda_q8_context *context,
    const void *device_input, void *device_output, size_t bytes) {
    (void)context;
    (void)device_input;
    (void)device_output;
    (void)bytes;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_add_device(fortai_cuda_q8_context *context,
    const void *device_left, const void *device_right, void *device_output,
    size_t elements) {
    (void)context;
    (void)device_left;
    (void)device_right;
    (void)device_output;
    (void)elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_rms_norm_device(fortai_cuda_q8_context *context,
    const void *device_input, const void *device_weights, void *device_output,
    size_t elements, float epsilon) {
    (void)context;
    (void)device_input;
    (void)device_weights;
    (void)device_output;
    (void)elements;
    (void)epsilon;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_silu_product_device(fortai_cuda_q8_context *context,
    void *device_gate, const void *device_up, size_t elements) {
    (void)context;
    (void)device_gate;
    (void)device_up;
    (void)elements;
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

int FORTAI_WEAK fortai_cuda_q8_matvec_host_pair(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *first_weights,
    const fortai_cuda_q8_weights *second_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *elapsed_ms) {
    (void)context;
    (void)first_weights;
    (void)second_weights;
    (void)host_activation;
    (void)activation_bytes;
    (void)first_output;
    (void)first_output_bytes;
    (void)second_output;
    (void)second_output_bytes;
    if (elapsed_ms) *elapsed_ms = 0.0f;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_matvec_host_triplet(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *first_weights,
    const fortai_cuda_q8_weights *second_weights,
    const fortai_cuda_q8_weights *third_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *third_output,
    size_t third_output_bytes, float *elapsed_ms) {
    (void)context;
    (void)first_weights;
    (void)second_weights;
    (void)third_weights;
    (void)host_activation;
    (void)activation_bytes;
    (void)first_output;
    (void)first_output_bytes;
    (void)second_output;
    (void)second_output_bytes;
    (void)third_output;
    (void)third_output_bytes;
    if (elapsed_ms) *elapsed_ms = 0.0f;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_matvec_host_triplet_contiguous(
    fortai_cuda_q8_context *context, const fortai_cuda_q8_weights *first_weights,
    const fortai_cuda_q8_weights *second_weights,
    const fortai_cuda_q8_weights *third_weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t host_output_bytes,
    float *elapsed_ms) {
    (void)context;
    (void)first_weights;
    (void)second_weights;
    (void)third_weights;
    (void)host_activation;
    (void)activation_bytes;
    (void)host_output;
    (void)host_output_bytes;
    if (elapsed_ms) *elapsed_ms = 0.0f;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_ffn_host(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *gate_weights,
    const fortai_cuda_q8_weights *up_weights,
    const fortai_cuda_q8_weights *down_weights, const void *host_activation,
    size_t activation_bytes, float *host_output, size_t output_bytes,
    float *elapsed_ms) {
    (void)context;
    (void)gate_weights;
    (void)up_weights;
    (void)down_weights;
    (void)host_activation;
    (void)activation_bytes;
    (void)host_output;
    (void)output_bytes;
    if (elapsed_ms) *elapsed_ms = 0.0f;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_q8_ffn_device(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *gate_weights,
    const fortai_cuda_q8_weights *up_weights,
    const fortai_cuda_q8_weights *down_weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    (void)context;
    (void)gate_weights;
    (void)up_weights;
    (void)down_weights;
    (void)device_activation;
    (void)activation_elements;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_recurrent_create(fortai_cuda_q8_context *context,
    const fortai_cuda_q8_weights *qkv_weights,
    const fortai_cuda_q8_weights *gate_weights,
    const fortai_cuda_q8_weights *alpha_weights,
    const fortai_cuda_q8_weights *beta_weights,
    const fortai_cuda_q8_weights *output_weights,
    const void *conv_weights, size_t conv_weight_bytes, int conv_size, int conv_kernel,
    const void *ssm_a, size_t ssm_a_bytes, const void *ssm_dt, size_t ssm_dt_bytes,
    const void *ssm_norm, size_t ssm_norm_bytes, int state_size, int key_heads,
    int value_heads, int head_size, int inner_size, float norm_epsilon,
    fortai_cuda_qwen35_recurrent **layer) {
    (void)context;
    (void)qkv_weights;
    (void)gate_weights;
    (void)alpha_weights;
    (void)beta_weights;
    (void)output_weights;
    (void)conv_weights;
    (void)conv_weight_bytes;
    (void)conv_size;
    (void)conv_kernel;
    (void)ssm_a;
    (void)ssm_a_bytes;
    (void)ssm_dt;
    (void)ssm_dt_bytes;
    (void)ssm_norm;
    (void)ssm_norm_bytes;
    (void)state_size;
    (void)key_heads;
    (void)value_heads;
    (void)head_size;
    (void)inner_size;
    (void)norm_epsilon;
    if (layer) *layer = NULL;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_recurrent_create_state(fortai_cuda_q8_context *context,
    const void *conv_weights, size_t conv_weight_bytes, int conv_size, int conv_kernel,
    const void *ssm_a, size_t ssm_a_bytes, const void *ssm_dt, size_t ssm_dt_bytes,
    const void *ssm_norm, size_t ssm_norm_bytes, int state_size, int key_heads,
    int value_heads, int head_size, int inner_size, float norm_epsilon,
    fortai_cuda_qwen35_recurrent **layer) {
    (void)context; (void)conv_weights; (void)conv_weight_bytes; (void)conv_size;
    (void)conv_kernel; (void)ssm_a; (void)ssm_a_bytes; (void)ssm_dt; (void)ssm_dt_bytes;
    (void)ssm_norm; (void)ssm_norm_bytes; (void)state_size; (void)key_heads;
    (void)value_heads; (void)head_size; (void)inner_size; (void)norm_epsilon;
    if (layer) *layer = NULL;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_recurrent_destroy(
    fortai_cuda_qwen35_recurrent *layer) {
    (void)layer;
    return FORTAI_CUDA_OK;
}

int FORTAI_WEAK fortai_cuda_qwen35_recurrent_reset(
    fortai_cuda_qwen35_recurrent *layer) {
    (void)layer;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_recurrent_run(fortai_cuda_qwen35_recurrent *layer,
    const void *host_activation, size_t activation_bytes, float *host_output,
    size_t output_bytes, float *elapsed_ms) {
    (void)layer;
    (void)host_activation;
    (void)activation_bytes;
    (void)host_output;
    (void)output_bytes;
    if (elapsed_ms) *elapsed_ms = 0.0f;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_recurrent_run_device(
    fortai_cuda_qwen35_recurrent *layer, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    (void)layer;
    (void)device_activation;
    (void)activation_elements;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_recurrent_run_core_device(
    fortai_cuda_qwen35_recurrent *layer, void *device_qkv, size_t qkv_elements,
    const void *device_gate, size_t gate_elements, const void *device_alpha,
    size_t alpha_elements, const void *device_beta, size_t beta_elements,
    void *device_output, size_t output_elements) {
    (void)layer; (void)device_qkv; (void)qkv_elements; (void)device_gate;
    (void)gate_elements; (void)device_alpha; (void)alpha_elements; (void)device_beta;
    (void)beta_elements; (void)device_output; (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_attention_create(
    fortai_cuda_q8_context *context, const fortai_cuda_q8_weights *query_weights,
    const fortai_cuda_q8_weights *key_weights, const fortai_cuda_q8_weights *value_weights,
    const fortai_cuda_q8_weights *output_weights, const void *query_norm,
    size_t query_norm_bytes, const void *key_norm, size_t key_norm_bytes,
    int heads, int key_value_heads, int head_size, int value_size, int max_context,
    int rope_dimension, float rope_base, float norm_epsilon,
    int cache_key_q8, int cache_value_q8,
    fortai_cuda_qwen35_attention **layer) {
    (void)context;
    (void)query_weights;
    (void)key_weights;
    (void)value_weights;
    (void)output_weights;
    (void)query_norm;
    (void)query_norm_bytes;
    (void)key_norm;
    (void)key_norm_bytes;
    (void)heads;
    (void)key_value_heads;
    (void)head_size;
    (void)value_size;
    (void)max_context;
    (void)rope_dimension;
    (void)rope_base;
    (void)norm_epsilon;
    (void)cache_key_q8;
    (void)cache_value_q8;
    if (layer) *layer = NULL;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_attention_create_state(
    fortai_cuda_q8_context *context, const void *query_norm, size_t query_norm_bytes,
    const void *key_norm, size_t key_norm_bytes, int heads, int key_value_heads,
    int head_size, int value_size, int max_context, int rope_dimension,
    float rope_base, float norm_epsilon, int cache_key_q8, int cache_value_q8,
    fortai_cuda_qwen35_attention **layer) {
    (void)context; (void)query_norm; (void)query_norm_bytes; (void)key_norm;
    (void)key_norm_bytes; (void)heads; (void)key_value_heads; (void)head_size;
    (void)value_size; (void)max_context; (void)rope_dimension; (void)rope_base;
    (void)norm_epsilon; (void)cache_key_q8; (void)cache_value_q8;
    if (layer) *layer = NULL;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_attention_destroy(
    fortai_cuda_qwen35_attention *layer) {
    (void)layer;
    return FORTAI_CUDA_OK;
}

int FORTAI_WEAK fortai_cuda_qwen35_attention_reset(
    fortai_cuda_qwen35_attention *layer) {
    (void)layer;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_attention_run_device(
    fortai_cuda_qwen35_attention *layer, const void *device_activation,
    size_t activation_elements, int position, void *device_output, size_t output_elements) {
    (void)layer;
    (void)device_activation;
    (void)activation_elements;
    (void)position;
    (void)device_output;
    (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_attention_run_core_device(
    fortai_cuda_qwen35_attention *layer, const void *device_query,
    size_t query_elements, const void *device_key, size_t key_elements,
    const void *device_value, size_t value_elements, int position,
    void *device_output, size_t output_elements) {
    (void)layer; (void)device_query; (void)query_elements; (void)device_key;
    (void)key_elements; (void)device_value; (void)value_elements; (void)position;
    (void)device_output; (void)output_elements;
    return unavailable();
}

int FORTAI_WEAK fortai_cuda_qwen35_argmax_device(
    fortai_cuda_q8_context *context, const void *device_logits,
    size_t elements, int *host_index) {
    (void)context;
    (void)device_logits;
    (void)elements;
    if (host_index) *host_index = 0;
    return unavailable();
}

const char * FORTAI_WEAK fortai_cuda_q8_last_error(
    const fortai_cuda_q8_context *context) {
    (void)context;
    return "CUDA backend is not linked";
}
