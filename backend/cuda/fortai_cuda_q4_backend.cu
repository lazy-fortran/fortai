#include "fortai_cuda_q4_backend.h"

#include <chrono>
#include <new>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

namespace {

// GGML's CUDA scheduler is deliberately chatty while it captures and reuses
// graphs.  FortAI owns the outer benchmark/runner protocol, so forwarding
// those informational messages would corrupt its machine-readable output.
void fortai_quiet_ggml_log(enum ggml_log_level, const char *, void *) {
}

ggml_log_callback fortai_previous_log;
void * fortai_previous_log_data = nullptr;
int fortai_quiet_log_users = 0;

void fortai_quiet_log_acquire() {
    if (fortai_quiet_log_users++ == 0) {
        ggml_log_get(&fortai_previous_log, &fortai_previous_log_data);
        ggml_log_set(fortai_quiet_ggml_log, nullptr);
    }
}

void fortai_quiet_log_release() {
    if (fortai_quiet_log_users <= 0) return;
    if (--fortai_quiet_log_users == 0) {
        ggml_log_set(fortai_previous_log, fortai_previous_log_data);
        fortai_previous_log = nullptr;
        fortai_previous_log_data = nullptr;
    }
}

bool fortai_supported_type(int type) {
    switch (type) {
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ4_XS:
        return true;
    default:
        return false;
    }
}

} // namespace

struct fortai_cuda_q4_context {
    ggml_backend_t devices[2] = {nullptr, nullptr};
    ggml_backend_t cpu = nullptr;
};

struct fortai_cuda_q4_weights {
    fortai_cuda_q4_context * owner = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_context * graph_context = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * activation = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    size_t output_bytes = 0;
};

static int fail(fortai_cuda_q4_context * context) {
    (void) context;
    return FORTAI_CUDA_RUNTIME_ERROR;
}

int fortai_cuda_q4_context_create(int first_device, int second_device,
    fortai_cuda_q4_context **out) {
    if (out == nullptr || first_device < 0 || second_device < 0) return FORTAI_CUDA_INVALID;
    *out = nullptr;
    auto * created = new (std::nothrow) fortai_cuda_q4_context;
    if (created == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
    fortai_quiet_log_acquire();
    created->devices[0] = ggml_backend_cuda_init(first_device);
    created->devices[1] = ggml_backend_cuda_init(second_device);
    created->cpu = ggml_backend_cpu_init();
    if (created->devices[0] == nullptr || created->devices[1] == nullptr || created->cpu == nullptr) {
        fortai_cuda_q4_context_destroy(created);
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    *out = created;
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_context_destroy(fortai_cuda_q4_context * context) {
    if (context == nullptr) return FORTAI_CUDA_OK;
    if (context->devices[0] != nullptr) ggml_backend_free(context->devices[0]);
    if (context->devices[1] != nullptr) ggml_backend_free(context->devices[1]);
    if (context->cpu != nullptr) ggml_backend_free(context->cpu);
    delete context;
    fortai_quiet_log_release();
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_weights_upload(fortai_cuda_q4_context * context, int value_type,
    const void * host_weights, size_t weight_bytes, int rows, int width, int device,
    fortai_cuda_q4_weights **out) {
    if (out == nullptr || context == nullptr || host_weights == nullptr || rows <= 0 || width <= 0 ||
        device < 0 || device > 1 || !fortai_supported_type(value_type)) {
        return FORTAI_CUDA_INVALID;
    }
    *out = nullptr;
    auto * created = new (std::nothrow) fortai_cuda_q4_weights;
    if (created == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
    created->owner = context;
    created->backend = context->devices[device];
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = true;
    created->graph_context = ggml_init(params);
    if (created->graph_context == nullptr) {
        fortai_cuda_q4_weights_destroy(created);
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    created->weight = ggml_new_tensor_2d(created->graph_context,
        static_cast<ggml_type>(value_type), width, rows);
    created->activation = ggml_new_tensor_2d(created->graph_context, GGML_TYPE_F32, width, 1);
    created->output = ggml_mul_mat(created->graph_context, created->weight, created->activation);
    created->graph = ggml_new_graph_custom(created->graph_context, GGML_DEFAULT_GRAPH_SIZE, false);
    if (created->weight == nullptr || created->activation == nullptr || created->output == nullptr ||
        created->graph == nullptr || ggml_row_size(static_cast<ggml_type>(value_type), width) * rows != weight_bytes) {
        fortai_cuda_q4_weights_destroy(created);
        return FORTAI_CUDA_INVALID;
    }
    ggml_build_forward_expand(created->graph, created->output);
    ggml_backend_t backends[] = {context->devices[device], context->cpu};
    created->scheduler = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (created->scheduler == nullptr || !ggml_backend_sched_alloc_graph(created->scheduler, created->graph)) {
        fortai_cuda_q4_weights_destroy(created);
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    ggml_backend_tensor_set(created->weight, host_weights, 0, weight_bytes);
    created->output_bytes = static_cast<size_t>(rows) * sizeof(float);
    *out = created;
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_weights_destroy(fortai_cuda_q4_weights * weights) {
    if (weights == nullptr) return FORTAI_CUDA_OK;
    if (weights->scheduler != nullptr) ggml_backend_sched_free(weights->scheduler);
    if (weights->graph_context != nullptr) ggml_free(weights->graph_context);
    delete weights;
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_matvec_host(fortai_cuda_q4_context * context,
    const fortai_cuda_q4_weights * weights, const void * host_activation,
    size_t activation_bytes, float * host_output, size_t output_bytes, float * elapsed_ms) {
    if (context == nullptr || weights == nullptr || weights->owner != context || host_activation == nullptr ||
        host_output == nullptr || activation_bytes != static_cast<size_t>(weights->activation->ne[0]) * sizeof(float) ||
        output_bytes != weights->output_bytes) return FORTAI_CUDA_INVALID;
    auto start = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(weights->activation, host_activation, 0, activation_bytes);
    enum ggml_status status = ggml_backend_sched_graph_compute(weights->scheduler, weights->graph);
    if (status != GGML_STATUS_SUCCESS) return fail(context);
    ggml_backend_tensor_get(weights->output, host_output, 0, output_bytes);
    ggml_backend_synchronize(weights->backend);
    auto stop = std::chrono::steady_clock::now();
    if (elapsed_ms != nullptr) *elapsed_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    return FORTAI_CUDA_OK;
}
