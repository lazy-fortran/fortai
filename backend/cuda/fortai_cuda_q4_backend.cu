#include "fortai_cuda_q4_backend.h"

#include <algorithm>
#include <chrono>
#include <new>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>
#include <cuda_runtime.h>

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
    int device_ids[2] = {0, 0};
    void *bridge_activation[2] = {nullptr, nullptr};
    void *bridge_output[2] = {nullptr, nullptr};
    size_t bridge_activation_bytes[2] = {0, 0};
    size_t bridge_output_bytes[2] = {0, 0};
    std::vector<fortai_cuda_q4_group_plan> group_plans;
};

/* Keep the externally rebound edge pointers stable for the lifetime of the
 * context.  GGML-CUDA may retain a CUDA-graph executable for a per-weight
 * graph; growing a bridge buffer after that graph has been warmed would
 * invalidate the captured pointer even though the tensor descriptor is later
 * rebound.  These reserves cover the largest single-token activation and a
 * three-way projection group in the supported Qwen3.8 models. */
constexpr size_t q4_bridge_activation_reserve = 1u << 20;
constexpr size_t q4_bridge_output_reserve = 1u << 24;

struct fortai_cuda_q4_weights {
    fortai_cuda_q4_context * owner = nullptr;
    ggml_backend_t backend = nullptr;
    int device = 0;
    ggml_context * graph_context = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * activation = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    size_t output_bytes = 0;
    ggml_context *embedding_context = nullptr;
    ggml_tensor *embedding_weight = nullptr;
    ggml_tensor *embedding_ids = nullptr;
    ggml_tensor *embedding_output = nullptr;
    ggml_cgraph *embedding_graph = nullptr;
    ggml_backend_buffer_t embedding_buffer = nullptr;
    bool embedding_ready = false;
};

static int fail(fortai_cuda_q4_context * context) {
    (void) context;
    return FORTAI_CUDA_RUNTIME_ERROR;
}

/* GGML owns its CUDA streams, while FortAI launches the following kernels on
 * its own streams.  A backend synchronize waits for GGML's work, but does not
 * establish a dependency for work subsequently submitted by the caller on a
 * different stream.  Establish that handoff explicitly before returning a
 * device result. */
static int fortai_cuda_q4_device_barrier(int device) {
    if (cudaSetDevice(device) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_context_create(int first_device, int second_device,
    fortai_cuda_q4_context **out) {
    if (out == nullptr || first_device < 0 || second_device < 0) return FORTAI_CUDA_INVALID;
    *out = nullptr;
    auto * created = new (std::nothrow) fortai_cuda_q4_context;
    if (created == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
    created->device_ids[0] = first_device;
    created->device_ids[1] = second_device;
    fortai_quiet_log_acquire();
    created->devices[0] = ggml_backend_cuda_init(first_device);
    created->devices[1] = second_device == first_device ? created->devices[0] :
        ggml_backend_cuda_init(second_device);
    created->cpu = ggml_backend_cpu_init();
    if (created->devices[0] == nullptr || created->devices[1] == nullptr || created->cpu == nullptr) {
        fortai_cuda_q4_context_destroy(created);
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    if (first_device != second_device) {
        cudaSetDevice(second_device);
        if (cudaMalloc(&created->bridge_activation[1], q4_bridge_activation_reserve) != cudaSuccess ||
            cudaMalloc(&created->bridge_output[1], q4_bridge_output_reserve) != cudaSuccess) {
            fortai_cuda_q4_context_destroy(created);
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        created->bridge_activation_bytes[1] = q4_bridge_activation_reserve;
        created->bridge_output_bytes[1] = q4_bridge_output_reserve;
    }
    *out = created;
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_context_destroy(fortai_cuda_q4_context * context) {
    if (context == nullptr) return FORTAI_CUDA_OK;
    for (int i = 0; i < 2; ++i) {
        cudaSetDevice(context->device_ids[i]);
        cudaFree(context->bridge_activation[i]);
        cudaFree(context->bridge_output[i]);
    }
    if (context->devices[0] != nullptr) ggml_backend_free(context->devices[0]);
    if (context->devices[1] != nullptr && context->devices[1] != context->devices[0])
        ggml_backend_free(context->devices[1]);
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
    created->device = context->device_ids[device];
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
    created->buffer = ggml_backend_alloc_ctx_tensors(created->graph_context, created->backend);
    if (created->buffer == nullptr) {
        fortai_cuda_q4_weights_destroy(created);
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    ggml_backend_buffer_set_usage(created->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
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
    if (weights->embedding_buffer != nullptr)
        ggml_backend_buffer_free(weights->embedding_buffer);
    if (weights->embedding_context != nullptr)
        ggml_free(weights->embedding_context);
    if (weights->buffer != nullptr) ggml_backend_buffer_free(weights->buffer);
    if (weights->graph_context != nullptr) ggml_free(weights->graph_context);
    delete weights;
    return FORTAI_CUDA_OK;
}

static int fortai_cuda_q4_prepare_embedding(fortai_cuda_q4_weights *weights) {
    if (weights == nullptr || weights->graph_context == nullptr || weights->weight == nullptr)
        return FORTAI_CUDA_INVALID;
    if (weights->embedding_ready) return FORTAI_CUDA_OK;
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = true;
    weights->embedding_context = ggml_init(params);
    if (weights->embedding_context == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
    weights->embedding_weight = ggml_new_tensor_2d(weights->embedding_context,
        weights->weight->type, weights->weight->ne[0], weights->weight->ne[1]);
    weights->embedding_ids = ggml_new_tensor_1d(weights->embedding_context, GGML_TYPE_I32, 1);
    if (weights->embedding_weight == nullptr || weights->embedding_ids == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
    /* Reuse the already-uploaded quantized weight allocation.  The auxiliary
     * context owns only the tiny id/output tensors below. */
    weights->embedding_weight->data = weights->weight->data;
    weights->embedding_weight->buffer = weights->weight->buffer;
    weights->embedding_output = ggml_get_rows(weights->embedding_context,
        weights->embedding_weight, weights->embedding_ids);
    weights->embedding_graph = ggml_new_graph_custom(weights->embedding_context,
        GGML_DEFAULT_GRAPH_SIZE, false);
    if (weights->embedding_output == nullptr || weights->embedding_graph == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
    ggml_build_forward_expand(weights->embedding_graph, weights->embedding_output);
    weights->embedding_buffer = ggml_backend_alloc_ctx_tensors(weights->embedding_context, weights->backend);
    if (weights->embedding_buffer == nullptr || weights->embedding_ids->data == nullptr ||
        weights->embedding_output->data == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
    weights->embedding_ready = true;
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
            enum ggml_status status = ggml_backend_graph_compute_async(
                weight->backend, weight->graph);
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

static int fortai_cuda_q4_ensure_bridge(fortai_cuda_q4_context *context, int slot,
    size_t activation_bytes, size_t output_bytes) {
    if (slot < 0 || slot > 1) return FORTAI_CUDA_INVALID;
    const int device = context->device_ids[slot];
    cudaSetDevice(device);
    if (activation_bytes > context->bridge_activation_bytes[slot]) {
        cudaFree(context->bridge_activation[slot]);
        context->bridge_activation[slot] = nullptr;
        if (cudaMalloc(&context->bridge_activation[slot], activation_bytes) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        context->bridge_activation_bytes[slot] = activation_bytes;
    }
    if (output_bytes > context->bridge_output_bytes[slot]) {
        cudaFree(context->bridge_output[slot]);
        context->bridge_output[slot] = nullptr;
        if (cudaMalloc(&context->bridge_output[slot], output_bytes) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        context->bridge_output_bytes[slot] = output_bytes;
    }
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_matvec_device(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    if (context == nullptr || weights == nullptr || weights->owner != context ||
        device_activation == nullptr || device_output == nullptr || activation_elements == 0 ||
        output_elements < static_cast<size_t>(weights->output->ne[0]))
        return FORTAI_CUDA_INVALID;

    /* GGML keeps the operation graph and its tensor descriptors separate from
     * the allocation itself.  Temporarily rebinding the two F32 edge tensors
     * lets the CUDA backend consume/produce FortAI's resident buffers without
     * staging through pageable host memory.  Restore the descriptors before
     * returning so the existing host API remains valid for fallback paths. */
    if (activation_elements != static_cast<size_t>(weights->activation->ne[0]))
        return FORTAI_CUDA_INVALID;
    const size_t activation_bytes = activation_elements * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(weights->output->ne[0]) * sizeof(float);
    const int primary_device = context->device_ids[0];
    const bool remote = weights->device != primary_device;
    int bridge_slot = -1;
    void *activation_ptr = const_cast<void *>(device_activation);
    void *output_ptr = device_output;
    if (remote) {
        if (weights->device != context->device_ids[1]) return FORTAI_CUDA_INVALID;
        bridge_slot = 1;
        int code = fortai_cuda_q4_ensure_bridge(context, bridge_slot, activation_bytes, output_bytes);
        if (code != FORTAI_CUDA_OK) return code;
        cudaSetDevice(primary_device);
        if (cudaMemcpyPeer(context->bridge_activation[bridge_slot], weights->device,
                device_activation, primary_device, activation_bytes) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        activation_ptr = context->bridge_activation[bridge_slot];
        output_ptr = context->bridge_output[bridge_slot];
    }
    void *saved_activation = weights->activation->data;
    void *saved_output = weights->output->data;
    weights->activation->data = activation_ptr;
    weights->output->data = output_ptr;
    enum ggml_status status = ggml_backend_graph_compute_async(weights->backend, weights->graph);
    if (status == GGML_STATUS_SUCCESS)
        ggml_backend_synchronize(weights->backend);
    if (status == GGML_STATUS_SUCCESS && fortai_cuda_q4_device_barrier(weights->device) != FORTAI_CUDA_OK)
        status = GGML_STATUS_FAILED;
    weights->activation->data = saved_activation;
    weights->output->data = saved_output;
    if (status == GGML_STATUS_SUCCESS && remote) {
        cudaSetDevice(primary_device);
        if (cudaMemcpyPeer(device_output, primary_device, output_ptr, weights->device, output_bytes) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
    }
    if (status == GGML_STATUS_SUCCESS && fortai_cuda_q4_device_barrier(primary_device) != FORTAI_CUDA_OK)
        return FORTAI_CUDA_RUNTIME_ERROR;
    return status == GGML_STATUS_SUCCESS ? FORTAI_CUDA_OK : FORTAI_CUDA_RUNTIME_ERROR;
}

static fortai_cuda_q4_group_plan * fortai_cuda_q4_find_device_group_plan(
    fortai_cuda_q4_context *context, const fortai_cuda_q4_weights * const *weights,
    const int *members, int member_count) {
    const fortai_cuda_q4_weights *first = weights[members[0]];
    for (auto & candidate : context->group_plans) {
        if (candidate.count != member_count || candidate.backend != first->backend) continue;
        bool match = true;
        for (int i = 0; i < member_count; ++i)
            if (candidate.weights[i] != weights[members[i]]) match = false;
        if (match) return &candidate;
    }

    fortai_cuda_q4_group_plan candidate;
    candidate.count = member_count;
    candidate.backend = first->backend;
    candidate.context = first->graph_context;
    candidate.graph = ggml_new_graph_custom(candidate.context, GGML_DEFAULT_GRAPH_SIZE, false);
    if (candidate.graph == nullptr) return nullptr;
    for (int i = 0; i < member_count; ++i) {
        candidate.weights[i] = weights[members[i]];
        ggml_build_forward_expand(candidate.graph, candidate.weights[i]->output);
    }
    context->group_plans.push_back(candidate);
    return &context->group_plans.back();
}

int fortai_cuda_q4_matvec_device_group(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights * const *weights, const void *device_activation,
    size_t activation_elements, void * const *device_outputs,
    const size_t *output_elements, int count) {
    if (context == nullptr || weights == nullptr || device_activation == nullptr ||
        device_outputs == nullptr || output_elements == nullptr || count < 1 || count > 3 ||
        activation_elements == 0) return FORTAI_CUDA_INVALID;

    const size_t activation_bytes = activation_elements * sizeof(float);
    const int primary_device = context->device_ids[0];
    for (int i = 0; i < count; ++i) {
        if (weights[i] == nullptr || weights[i]->owner != context || device_outputs[i] == nullptr ||
            output_elements[i] < static_cast<size_t>(weights[i]->output->ne[0]) ||
            activation_elements != static_cast<size_t>(weights[i]->activation->ne[0]))
            return FORTAI_CUDA_INVALID;
    }

    bool done[3] = {false, false, false};
    for (int first_index = 0; first_index < count; ++first_index) {
        if (done[first_index]) continue;
        const int device = weights[first_index]->device;
        if (device != primary_device && device != context->device_ids[1])
            return FORTAI_CUDA_INVALID;

        int members[3] = {0, 0, 0};
        int member_count = 0;
        for (int i = first_index; i < count; ++i) {
            if (!done[i] && weights[i]->device == device)
                members[member_count++] = i;
        }
        if (member_count == 0) continue;
        for (int i = 0; i < member_count; ++i) done[members[i]] = true;

        const bool remote = device != primary_device;
        void * activation_ptr = const_cast<void *>(device_activation);
        size_t bridge_output_bytes = 0;
        size_t output_offsets[3] = {0, 0, 0};
        for (int i = 0; i < member_count; ++i) {
            output_offsets[i] = bridge_output_bytes;
            bridge_output_bytes += static_cast<size_t>(weights[members[i]]->output->ne[0]) * sizeof(float);
        }
        if (remote) {
            int code = fortai_cuda_q4_ensure_bridge(context, 1, activation_bytes, bridge_output_bytes);
            if (code != FORTAI_CUDA_OK) return code;
            cudaSetDevice(primary_device);
            if (cudaMemcpyPeer(context->bridge_activation[1], device,
                    device_activation, primary_device, activation_bytes) != cudaSuccess)
                return FORTAI_CUDA_RUNTIME_ERROR;
            activation_ptr = context->bridge_activation[1];
        }

        fortai_cuda_q4_group_plan *plan =
            fortai_cuda_q4_find_device_group_plan(context, weights, members, member_count);
        if (plan == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
        void *saved_activation[3] = {nullptr, nullptr, nullptr};
        void *saved_output[3] = {nullptr, nullptr, nullptr};
        for (int i = 0; i < member_count; ++i) {
            const int index = members[i];
            saved_activation[i] = weights[index]->activation->data;
            saved_output[i] = weights[index]->output->data;
            weights[index]->activation->data = activation_ptr;
            weights[index]->output->data = remote ?
                static_cast<void *>(static_cast<char *>(context->bridge_output[1]) + output_offsets[i]) :
                device_outputs[index];
        }
        enum ggml_status status = ggml_backend_graph_compute_async(plan->backend, plan->graph);
        if (status == GGML_STATUS_SUCCESS) ggml_backend_synchronize(plan->backend);
        if (status == GGML_STATUS_SUCCESS && fortai_cuda_q4_device_barrier(device) != FORTAI_CUDA_OK)
            status = GGML_STATUS_FAILED;
        for (int i = 0; i < member_count; ++i) {
            const int index = members[i];
            weights[index]->activation->data = saved_activation[i];
            weights[index]->output->data = saved_output[i];
        }
        if (status != GGML_STATUS_SUCCESS) return FORTAI_CUDA_RUNTIME_ERROR;
        if (remote) {
            cudaSetDevice(primary_device);
            for (int i = 0; i < member_count; ++i) {
                const int index = members[i];
                const size_t bytes = static_cast<size_t>(weights[index]->output->ne[0]) * sizeof(float);
                if (cudaMemcpyPeer(device_outputs[index], primary_device,
                        static_cast<char *>(context->bridge_output[1]) + output_offsets[i], device,
                        bytes) != cudaSuccess) return FORTAI_CUDA_RUNTIME_ERROR;
            }
        }
        if (fortai_cuda_q4_device_barrier(primary_device) != FORTAI_CUDA_OK)
            return FORTAI_CUDA_RUNTIME_ERROR;
    }
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_embedding_device(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, int64_t token_id,
    void *device_output, size_t output_elements) {
    if (context == nullptr || weights == nullptr || weights->owner != context ||
        device_output == nullptr || token_id < 0 || token_id >= weights->weight->ne[1] ||
        output_elements < static_cast<size_t>(weights->weight->ne[0]))
        return FORTAI_CUDA_INVALID;
    auto *mutable_weights = const_cast<fortai_cuda_q4_weights *>(weights);
    int code = fortai_cuda_q4_prepare_embedding(mutable_weights);
    if (code != FORTAI_CUDA_OK) return code;
    const int32_t id = static_cast<int32_t>(token_id);
    ggml_backend_tensor_set_async(weights->backend, weights->embedding_ids, &id, 0, sizeof(id));
    const int primary_device = context->device_ids[0];
    const bool remote = weights->device != primary_device;
    void *output_ptr = device_output;
    if (remote) {
        if (weights->device != context->device_ids[1]) return FORTAI_CUDA_INVALID;
        code = fortai_cuda_q4_ensure_bridge(context, 1,
            0, static_cast<size_t>(weights->weight->ne[0]) * sizeof(float));
        if (code != FORTAI_CUDA_OK) return code;
        output_ptr = context->bridge_output[1];
    }
    void *saved_output = weights->embedding_output->data;
    weights->embedding_output->data = output_ptr;
    enum ggml_status status = ggml_backend_graph_compute_async(weights->backend, weights->embedding_graph);
    if (status == GGML_STATUS_SUCCESS)
        ggml_backend_synchronize(weights->backend);
    if (status == GGML_STATUS_SUCCESS && fortai_cuda_q4_device_barrier(weights->device) != FORTAI_CUDA_OK)
        status = GGML_STATUS_FAILED;
    weights->embedding_output->data = saved_output;
    if (status == GGML_STATUS_SUCCESS && remote) {
        cudaSetDevice(primary_device);
        if (cudaMemcpyPeer(device_output, primary_device, output_ptr, weights->device,
                static_cast<size_t>(weights->weight->ne[0]) * sizeof(float)) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
    }
    if (status == GGML_STATUS_SUCCESS && fortai_cuda_q4_device_barrier(primary_device) != FORTAI_CUDA_OK)
        return FORTAI_CUDA_RUNTIME_ERROR;
    return status == GGML_STATUS_SUCCESS ? FORTAI_CUDA_OK : FORTAI_CUDA_RUNTIME_ERROR;
}
