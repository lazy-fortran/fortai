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

struct fortai_cuda_q4_weights;

struct fortai_cuda_q4_group_plan {
    int count = 0;
    const fortai_cuda_q4_weights *weights[3] = {nullptr, nullptr, nullptr};
    ggml_context *context = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_backend_t backend = nullptr;
};

struct fortai_cuda_q4_context {
    ggml_backend_t devices[2] = {nullptr, nullptr};
    ggml_backend_t cpu = nullptr;
    std::vector<fortai_cuda_q4_group_plan> group_plans;
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
    /* Queue model uploads on each device and synchronize once after the full
     * tensor set is built.  The previous per-tensor synchronous copy made
     * large UD-Q4_K_XL model open time scale with the tensor count. */
    ggml_backend_tensor_set_async(created->backend, created->weight, host_weights, 0, weight_bytes);
    created->output_bytes = static_cast<size_t>(rows) * sizeof(float);
    *out = created;
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_context_synchronize(fortai_cuda_q4_context *context) {
    if (context == nullptr || context->devices[0] == nullptr || context->devices[1] == nullptr)
        return FORTAI_CUDA_INVALID;
    ggml_backend_synchronize(context->devices[0]);
    if (context->devices[1] != context->devices[0]) ggml_backend_synchronize(context->devices[1]);
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_weights_destroy(fortai_cuda_q4_weights * weights) {
    if (weights == nullptr) return FORTAI_CUDA_OK;
    if (weights->scheduler != nullptr) ggml_backend_sched_free(weights->scheduler);
    if (weights->graph_context != nullptr) ggml_free(weights->graph_context);
    delete weights;
    return FORTAI_CUDA_OK;
}

static int fortai_cuda_q4_matvec_host_group(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights * const *weights, const void *host_activation,
    size_t activation_bytes, float * const *host_outputs, const size_t *output_bytes,
    int count, float *elapsed_ms) {
    if (context == nullptr || weights == nullptr || host_activation == nullptr ||
        host_outputs == nullptr || output_bytes == nullptr || count < 1 || count > 3)
        return FORTAI_CUDA_INVALID;
    for (int i = 0; i < count; ++i) {
        if (weights[i] == nullptr || weights[i]->owner != context || host_outputs[i] == nullptr ||
            activation_bytes != static_cast<size_t>(weights[i]->activation->ne[0]) * sizeof(float) ||
            output_bytes[i] != weights[i]->output_bytes) return FORTAI_CUDA_INVALID;
    }
    auto start = std::chrono::steady_clock::now();
    /* The synchronous wrapper used to force a host-visible boundary after
     * every projection.  Queue all work first; projections on different
     * devices can now overlap and projections on one device have no host gap
     * between their copies and kernels. */
    for (int i = 0; i < count; ++i) {
        const fortai_cuda_q4_weights *weight = weights[i];
        bool already_queued = false;
        for (int j = 0; j < i; ++j)
            if (weights[j]->backend == weight->backend) already_queued = true;
        if (already_queued) continue;

        int members[3];
        int member_count = 0;
        for (int j = i; j < count; ++j)
            if (weights[j]->backend == weight->backend) members[member_count++] = j;
        if (member_count == 1) {
            ggml_backend_tensor_set_async(weight->backend, weight->activation,
                host_activation, 0, activation_bytes);
            enum ggml_status status = ggml_backend_sched_graph_compute_async(
                weight->scheduler, weight->graph);
            if (status != GGML_STATUS_SUCCESS) return fail(context);
            ggml_backend_tensor_get_async(weight->backend, weight->output,
                host_outputs[i], 0, output_bytes[i]);
            continue;
        }

        fortai_cuda_q4_group_plan *plan = nullptr;
        for (auto &candidate : context->group_plans) {
            if (candidate.count != member_count || candidate.backend != weight->backend) continue;
            bool match = true;
            for (int k = 0; k < member_count; ++k)
                if (candidate.weights[k] != weights[members[k]]) match = false;
            if (match) {
                plan = &candidate;
                break;
            }
        }
        if (plan == nullptr) {
            fortai_cuda_q4_group_plan candidate;
            candidate.count = member_count;
            candidate.backend = weight->backend;
            candidate.context = weight->graph_context;
            for (int k = 0; k < member_count; ++k)
                candidate.weights[k] = weights[members[k]];
            candidate.graph = ggml_new_graph_custom(candidate.context, GGML_DEFAULT_GRAPH_SIZE, false);
            if (candidate.graph == nullptr) return fail(context);
            for (int k = 0; k < member_count; ++k)
                ggml_build_forward_expand(candidate.graph, candidate.weights[k]->output);
            context->group_plans.push_back(candidate);
            plan = &context->group_plans.back();
        }
        for (int k = 0; k < member_count; ++k) {
            const fortai_cuda_q4_weights *member = plan->weights[k];
            ggml_backend_tensor_set_async(member->backend, member->activation,
                host_activation, 0, activation_bytes);
        }
        enum ggml_status status = ggml_backend_graph_compute_async(plan->backend, plan->graph);
        if (status != GGML_STATUS_SUCCESS) return fail(context);
        for (int k = 0; k < member_count; ++k) {
            int output_index = members[k];
            ggml_backend_tensor_get_async(weight->backend, plan->weights[k]->output,
                host_outputs[output_index], 0, output_bytes[output_index]);
        }
    }
    /* A group may contain multiple per-weight schedulers, but schedulers on
     * the same CUDA backend share the backend's stream pool.  Synchronizing
     * the backend once avoids one host-side scheduler barrier per projection;
     * retain one barrier per participating device for multi-GPU groups. */
    for (int i = 0; i < count; ++i) {
        bool already_synchronized = false;
        for (int j = 0; j < i; ++j)
            if (weights[j]->backend == weights[i]->backend) already_synchronized = true;
        if (!already_synchronized) ggml_backend_synchronize(weights[i]->backend);
    }
    auto stop = std::chrono::steady_clock::now();
    if (elapsed_ms != nullptr)
        *elapsed_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_matvec_host(fortai_cuda_q4_context * context,
    const fortai_cuda_q4_weights * weights, const void * host_activation,
    size_t activation_bytes, float * host_output, size_t output_bytes, float * elapsed_ms) {
    const fortai_cuda_q4_weights *group[1] = {weights};
    float *outputs[1] = {host_output};
    const size_t bytes[1] = {output_bytes};
    return fortai_cuda_q4_matvec_host_group(context, group, host_activation,
        activation_bytes, outputs, bytes, 1, elapsed_ms);
}

int fortai_cuda_q4_matvec_host_pair(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *first_weights,
    const fortai_cuda_q4_weights *second_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *elapsed_ms) {
    const fortai_cuda_q4_weights *group[2] = {first_weights, second_weights};
    float *outputs[2] = {first_output, second_output};
    const size_t bytes[2] = {first_output_bytes, second_output_bytes};
    return fortai_cuda_q4_matvec_host_group(context, group, host_activation,
        activation_bytes, outputs, bytes, 2, elapsed_ms);
}

int fortai_cuda_q4_matvec_host_triplet(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *first_weights,
    const fortai_cuda_q4_weights *second_weights,
    const fortai_cuda_q4_weights *third_weights, const void *host_activation,
    size_t activation_bytes, float *first_output, size_t first_output_bytes,
    float *second_output, size_t second_output_bytes, float *third_output,
    size_t third_output_bytes, float *elapsed_ms) {
    const fortai_cuda_q4_weights *group[3] = {first_weights, second_weights, third_weights};
    float *outputs[3] = {first_output, second_output, third_output};
    const size_t bytes[3] = {first_output_bytes, second_output_bytes, third_output_bytes};
    return fortai_cuda_q4_matvec_host_group(context, group, host_activation,
        activation_bytes, outputs, bytes, 3, elapsed_ms);
}
