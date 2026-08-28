#include "fortai_cuda_q4_backend.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>
#include <cuda_runtime.h>

/* The public GGML backend intentionally does not expose its scheduler
 * stream.  FortAI's unified-stream mode needs the exact stream used by the
 * backend, not a second stream plus a device fence.  This private definition
 * is compiled against the same llama.cpp source tree as the linked GGML
 * library (the build script supplies the include path and graph macro). */
#include "ggml-backend-impl.h"
#include "common.cuh"

namespace {

/* All resident Qwen projection results are below 1 MiB.  Giving every
 * pinned-ring slot this fixed capacity avoids reallocating a slot when the
 * first few QKV/FFN shapes differ, which would otherwise force an event
 * synchronization in the decode hot path. */
constexpr size_t q4_host_bridge_reserve = 1u << 20;

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

static bool fortai_cuda_q4_segment_graph_requested();

struct fortai_cuda_q4_weights;

struct fortai_cuda_q4_group_plan {
    int count = 0;
    const fortai_cuda_q4_weights *weights[8] = {};
    ggml_context *context = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_backend_t backend = nullptr;
    mutable cudaGraph_t cached_graph = nullptr;
    mutable cudaGraphExec_t cached_exec = nullptr;
    mutable bool graph_warmup = false;
    mutable bool graph_disabled = false;
};

struct fortai_cuda_q4_batch_plan {
    int count = 0;
    int batch = 0;
    const fortai_cuda_q4_weights *weights[3] = {nullptr, nullptr, nullptr};
    ggml_context *context = nullptr;
    ggml_tensor *weight[3] = {nullptr, nullptr, nullptr};
    ggml_tensor *activation[3] = {nullptr, nullptr, nullptr};
    ggml_tensor *output[3] = {nullptr, nullptr, nullptr};
    ggml_cgraph *graph = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    mutable cudaGraph_t cached_graph = nullptr;
    mutable cudaGraphExec_t cached_exec = nullptr;
    mutable bool graph_warmup = false;
    mutable bool graph_disabled = false;
};

struct fortai_cuda_q4_embedding_batch_plan {
    const fortai_cuda_q4_weights *weights = nullptr;
    int batch = 0;
    ggml_context *context = nullptr;
    ggml_tensor *weight = nullptr;
    ggml_tensor *ids = nullptr;
    ggml_tensor *output = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    mutable cudaGraph_t cached_graph = nullptr;
    mutable cudaGraphExec_t cached_exec = nullptr;
    mutable bool graph_warmup = false;
    mutable bool graph_disabled = false;
};

struct fortai_cuda_q4_swiglu_plan {
    const fortai_cuda_q4_weights *gate = nullptr;
    const fortai_cuda_q4_weights *up = nullptr;
    ggml_context *context = nullptr;
    ggml_tensor *activation = nullptr;
    ggml_tensor *gate_output = nullptr;
    ggml_tensor *up_output = nullptr;
    ggml_tensor *output = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t compute_buffer = nullptr;
    mutable cudaGraph_t cached_graph = nullptr;
    mutable cudaGraphExec_t cached_exec = nullptr;
    mutable bool graph_warmup = false;
    mutable bool graph_disabled = false;
};

/* Complete resident Q4 FFN graph.  Keeping the down projection in the same
 * graph removes one scheduler/graph launch at every transformer layer while
 * retaining GGML-CUDA's format-specific fused matmul kernels. */
struct fortai_cuda_q4_ffn_plan {
    const fortai_cuda_q4_weights *gate = nullptr;
    const fortai_cuda_q4_weights *up = nullptr;
    const fortai_cuda_q4_weights *down = nullptr;
    ggml_context *context = nullptr;
    ggml_tensor *activation = nullptr;
    ggml_tensor *gate_output = nullptr;
    ggml_tensor *up_output = nullptr;
    ggml_tensor *swiglu_output = nullptr;
    ggml_tensor *output = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t compute_buffer = nullptr;
    mutable cudaGraph_t cached_graph = nullptr;
    mutable cudaGraphExec_t cached_exec = nullptr;
    mutable bool graph_warmup = false;
    mutable bool graph_disabled = false;
};


struct fortai_cuda_q4_bridge_events {
    /* Each operation gets independent events for the source-ready, remote-
     * ready, remote-done, output-buffer-free, and primary-done edges.  CUDA
     * events are reusable, but only after their previous record has
     * completed; the bridge-slot allocator checks that condition on wrap. */
    ggml_backend_event_t source_ready = nullptr;
    ggml_backend_event_t remote_ready = nullptr;
    ggml_backend_event_t remote_done = nullptr;
    ggml_backend_event_t output_free = nullptr;
    ggml_backend_event_t primary_done = nullptr;
};

struct fortai_cuda_q4_host_bridge_slot {
    void *buffer = nullptr;
    size_t bytes = 0;
    cudaEvent_t ready_event = nullptr;
    cudaEvent_t free_event = nullptr;
    bool free_recorded = false;
};

struct fortai_cuda_q4_weight_arena {
    ggml_backend_buffer_t buffer = nullptr;
    size_t used = 0;
    size_t capacity = 0;
};

struct fortai_cuda_q4_context {
    /* A decode traverses hundreds of quantized projections.  Keep enough
     * reusable events in flight that the normal same-device path never has
     * to wait on the host merely to recycle an event. */
    static constexpr int event_slots = 1024;
    static constexpr int host_bridge_slots = 8;
    ggml_backend_t devices[2] = {nullptr, nullptr};
    ggml_backend_t cpu = nullptr;
    int device_ids[2] = {0, 0};
    bool segment_graph_requested = false;
    bool sync_bridge_requested = false;
    cudaStream_t consumer_stream[2] = {nullptr, nullptr};
    ggml_backend_event_t producer_event[2][event_slots] = {};
    unsigned producer_event_index[2] = {0, 0};
    void *bridge_activation[2] = {nullptr, nullptr};
    void *bridge_output[2] = {nullptr, nullptr};
    size_t bridge_activation_bytes[2] = {0, 0};
    size_t bridge_output_bytes[2] = {0, 0};
    void *host_bridge = nullptr;
    size_t host_bridge_bytes = 0;
    /* CUDA can stage a peer copy through the host even when the topology
     * advertises no direct peer access.  Keep the explicit pinned-ring bridge
     * as the default because it overlaps the two PCIe legs on PHB systems;
     * FORTAI_CUDA_Q4_USE_PEER=1 enables the single-runtime peer operation for
     * drivers where that path is faster. */
    bool use_peer_copy = false;
    cudaStream_t bridge_stream[2] = {nullptr, nullptr};
    fortai_cuda_q4_bridge_events bridge_events[event_slots] = {};
    std::vector<fortai_cuda_q4_bridge_events *> extra_bridge_events;
    unsigned bridge_event_index = 0;
    ggml_backend_event_t last_activation_done = nullptr;
    ggml_backend_event_t last_output_free = nullptr;
    fortai_cuda_q4_host_bridge_slot input_host[host_bridge_slots] = {};
    fortai_cuda_q4_host_bridge_slot output_host[host_bridge_slots] = {};
    unsigned input_host_index = 0;
    unsigned output_host_index = 0;
    /* Stable host storage for the one-word get-rows index.  The asynchronous
     * GGML tensor upload must not point at the stack frame of a decode call. */
    int32_t embedding_id = 0;
    std::vector<fortai_cuda_q4_weight_arena> weight_arenas[2];
    std::vector<fortai_cuda_q4_group_plan> group_plans;
    std::vector<fortai_cuda_q4_batch_plan> batch_plans;
    std::vector<fortai_cuda_q4_embedding_batch_plan> embedding_batch_plans;
    std::vector<fortai_cuda_q4_swiglu_plan> swiglu_plans;
    std::vector<fortai_cuda_q4_ffn_plan> ffn_plans;
};

/* The linked GGML build exposes two additional 2-D async transfer slots in
 * ggml_backend_i.  Its backend object is therefore 0x98 bytes and stores the
 * context pointer at offset 0x90, while older public headers describe the
 * shorter 0x88-byte object.  Keep this ABI view local so the stream hand-off
 * never interprets the device pointer as a CUDA context. */
struct fortai_ggml_backend_runtime_layout {
    ggml_guid_t guid;
    void *iface[16];
    ggml_backend_dev_t device;
    void *context;
};
static_assert(offsetof(fortai_ggml_backend_runtime_layout, context) == 0x90,
    "unexpected GGML backend ABI layout");

/* ggml_backend_event_t is intentionally opaque in the public API.  The CUDA
 * backend stores the native event handle in its context field; keep this
 * tiny layout local so we can record on FortAI's producer stream and then use
 * the public ggml backend event-wait/record operations for the scheduler
 * stream.  It is the same two-field layout used by ggml-backend-impl.h. */
struct fortai_ggml_backend_event_layout {
    ggml_backend_dev_t device;
    void *context;
};

static cudaEvent_t fortai_cuda_event_handle(ggml_backend_event_t event) {
    if (event == nullptr) return nullptr;
    return reinterpret_cast<cudaEvent_t>(
        static_cast<fortai_ggml_backend_event_layout *>(static_cast<void *>(event))->context);
}

static int fortai_cuda_q4_query_event(cudaEvent_t event, int device) {
    if (event == nullptr) return FORTAI_CUDA_INVALID;
    if (cudaSetDevice(device) != cudaSuccess) return FORTAI_CUDA_RUNTIME_ERROR;
    const cudaError_t query = cudaEventQuery(event);
    if (query == cudaErrorNotReady) {
        if (cudaEventSynchronize(event) != cudaSuccess) return FORTAI_CUDA_RUNTIME_ERROR;
    } else if (query != cudaSuccess) {
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    return FORTAI_CUDA_OK;
}

static int fortai_cuda_q4_event_ready(cudaEvent_t event, int device, bool *ready) {
    if (event == nullptr || ready == nullptr) return FORTAI_CUDA_INVALID;
    if (cudaSetDevice(device) != cudaSuccess) return FORTAI_CUDA_RUNTIME_ERROR;
    const cudaError_t query = cudaEventQuery(event);
    if (query == cudaSuccess) {
        *ready = true;
        return FORTAI_CUDA_OK;
    }
    if (query == cudaErrorNotReady) {
        *ready = false;
        return FORTAI_CUDA_OK;
    }
    return FORTAI_CUDA_RUNTIME_ERROR;
}

static void fortai_cuda_q4_free_bridge_events(fortai_cuda_q4_context *context,
    fortai_cuda_q4_bridge_events *events);

static int fortai_cuda_q4_create_bridge_events(fortai_cuda_q4_context *context,
    fortai_cuda_q4_bridge_events *events) {
    if (context == nullptr || events == nullptr) return FORTAI_CUDA_INVALID;
    cudaSetDevice(context->device_ids[0]);
    events->source_ready = ggml_backend_event_new(ggml_backend_get_device(context->devices[0]));
    events->primary_done = ggml_backend_event_new(ggml_backend_get_device(context->devices[0]));
    cudaSetDevice(context->device_ids[1]);
    events->remote_ready = ggml_backend_event_new(ggml_backend_get_device(context->devices[1]));
    events->remote_done = ggml_backend_event_new(ggml_backend_get_device(context->devices[1]));
    events->output_free = ggml_backend_event_new(ggml_backend_get_device(context->devices[1]));
    if (events->source_ready == nullptr || events->primary_done == nullptr ||
        events->remote_ready == nullptr || events->remote_done == nullptr ||
        events->output_free == nullptr) {
        fortai_cuda_q4_free_bridge_events(context, events);
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    return FORTAI_CUDA_OK;
}

static void fortai_cuda_q4_free_bridge_events(fortai_cuda_q4_context *context,
    fortai_cuda_q4_bridge_events *events) {
    if (context == nullptr || events == nullptr) return;
    cudaSetDevice(context->device_ids[0]);
    if (events->source_ready != nullptr) ggml_backend_event_free(events->source_ready);
    if (events->primary_done != nullptr) ggml_backend_event_free(events->primary_done);
    cudaSetDevice(context->device_ids[1]);
    if (events->remote_ready != nullptr) ggml_backend_event_free(events->remote_ready);
    if (events->remote_done != nullptr) ggml_backend_event_free(events->remote_done);
    if (events->output_free != nullptr) ggml_backend_event_free(events->output_free);
    *events = {};
}

/* Pick a bridge event set.  The normal decode pipeline is serialized by the
 * model's dependency graph, so wrapping this ring is rare.  If a caller has
 * queued more work than the ring can hold, synchronize only the event slot
 * being recycled; this keeps the fast path entirely device ordered. */
static int fortai_cuda_q4_next_bridge_events(fortai_cuda_q4_context *context,
    fortai_cuda_q4_bridge_events **events) {
    if (context == nullptr || events == nullptr || context->bridge_stream[0] == nullptr ||
        context->bridge_stream[1] == nullptr)
        return FORTAI_CUDA_INVALID;
    const unsigned sequence = context->bridge_event_index++;
    const unsigned index = sequence % fortai_cuda_q4_context::event_slots;
    auto *candidate = &context->bridge_events[index];
    bool available = true;
    /* Newly-created events have never been recorded.  A slot can only carry
     * an older operation after the ring wraps, so avoid a host-side
     * cudaEventQuery/cudaSetDevice pair on every projection. */
    if (sequence >= fortai_cuda_q4_context::event_slots) {
        const ggml_backend_event_t all[] = {
            candidate->source_ready, candidate->remote_ready, candidate->remote_done,
            candidate->output_free, candidate->primary_done
        };
        const int devices[] = {
            context->device_ids[0], context->device_ids[1], context->device_ids[1],
            context->device_ids[1], context->device_ids[0]
        };
        for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i) {
            bool ready = false;
            if (fortai_cuda_q4_event_ready(fortai_cuda_event_handle(all[i]), devices[i], &ready) !=
                    FORTAI_CUDA_OK)
                return FORTAI_CUDA_RUNTIME_ERROR;
            if (!ready) available = false;
        }
    }
    if (!available) {
        auto *fresh = new (std::nothrow) fortai_cuda_q4_bridge_events;
        if (fresh == nullptr || fortai_cuda_q4_create_bridge_events(context, fresh) != FORTAI_CUDA_OK) {
            delete fresh;
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        try {
            context->extra_bridge_events.push_back(fresh);
        } catch (...) {
            fortai_cuda_q4_free_bridge_events(context, fresh);
            delete fresh;
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        *events = fresh;
        return FORTAI_CUDA_OK;
    }
    *events = candidate;
    return FORTAI_CUDA_OK;
}

static int fortai_cuda_q4_ensure_host_buffer(fortai_cuda_q4_host_bridge_slot *slot,
    size_t bytes, int device) {
    if (slot == nullptr || bytes == 0) return FORTAI_CUDA_INVALID;
    if (bytes <= slot->bytes && slot->buffer != nullptr) return FORTAI_CUDA_OK;
    if (slot->free_recorded) {
        if (fortai_cuda_q4_query_event(slot->free_event, device) != FORTAI_CUDA_OK)
            return FORTAI_CUDA_RUNTIME_ERROR;
        slot->free_recorded = false;
    }
    if (slot->buffer != nullptr) {
        if (cudaFreeHost(slot->buffer) != cudaSuccess) return FORTAI_CUDA_RUNTIME_ERROR;
        slot->buffer = nullptr;
        slot->bytes = 0;
    }
    if (cudaHostAlloc(&slot->buffer, bytes, cudaHostAllocPortable) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    slot->bytes = bytes;
    return FORTAI_CUDA_OK;
}

static int fortai_cuda_q4_acquire_input_host(fortai_cuda_q4_context *context,
    size_t bytes, int *index) {
    if (context == nullptr || index == nullptr) return FORTAI_CUDA_INVALID;
    const unsigned selected = context->input_host_index++ % fortai_cuda_q4_context::host_bridge_slots;
    auto &slot = context->input_host[selected];
    if (fortai_cuda_q4_ensure_host_buffer(&slot, std::max(bytes, q4_host_bridge_reserve),
            context->device_ids[1]) != FORTAI_CUDA_OK)
        return FORTAI_CUDA_RUNTIME_ERROR;
    /* The previous use of this host buffer ended with the remote H2D copy.
     * Waiting on that event on the primary bridge stream prevents a new D2H
     * from overwriting bytes still consumed by the remote transfer. */
    if (slot.free_recorded) {
        if (cudaSetDevice(context->device_ids[0]) != cudaSuccess ||
            cudaStreamWaitEvent(context->bridge_stream[0], slot.free_event, 0) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        slot.free_recorded = false;
    }
    *index = static_cast<int>(selected);
    return FORTAI_CUDA_OK;
}

static int fortai_cuda_q4_acquire_output_host(fortai_cuda_q4_context *context,
    size_t bytes, int *index) {
    if (context == nullptr || index == nullptr) return FORTAI_CUDA_INVALID;
    const unsigned selected = context->output_host_index++ % fortai_cuda_q4_context::host_bridge_slots;
    auto &slot = context->output_host[selected];
    if (fortai_cuda_q4_ensure_host_buffer(&slot, std::max(bytes, q4_host_bridge_reserve),
            context->device_ids[0]) != FORTAI_CUDA_OK)
        return FORTAI_CUDA_RUNTIME_ERROR;
    /* The remote D2H must not reuse a host slot before the primary H2D that
     * consumed its previous contents has completed. */
    if (slot.free_recorded) {
        if (cudaSetDevice(context->device_ids[1]) != cudaSuccess ||
            cudaStreamWaitEvent(context->bridge_stream[1], slot.free_event, 0) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        slot.free_recorded = false;
    }
    *index = static_cast<int>(selected);
    return FORTAI_CUDA_OK;
}

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
    cudaGraph_t cached_graph = nullptr;
    cudaGraphExec_t cached_exec = nullptr;
    bool graph_warmup = false;
    bool graph_disabled = false;
    mutable cudaGraph_t embedding_cached_graph = nullptr;
    mutable cudaGraphExec_t embedding_cached_exec = nullptr;
    mutable bool embedding_graph_warmup = false;
    mutable bool embedding_graph_disabled = false;
};

/* Build the same { Q4 matmul, Q4 matmul, SwiGLU } shape that llama.cpp
 * presents to GGML-CUDA.  GGML's CUDA scheduler recognizes this graph and
 * dispatches its quantization-specific fused gate/up kernel, so no second
 * copy of every Q4 format's dequantization code is needed here.  The two
 * weight descriptors remain owned by their normal per-tensor contexts; only
 * the small activation/intermediate graph is owned by this plan. */
static fortai_cuda_q4_swiglu_plan * fortai_cuda_q4_find_swiglu_plan(
    fortai_cuda_q4_context *context, const fortai_cuda_q4_weights *gate,
    const fortai_cuda_q4_weights *up) {
    if (context == nullptr || gate == nullptr || up == nullptr || gate->owner != context ||
        up->owner != context || gate->backend != up->backend || gate->device != up->device ||
        gate->weight == nullptr || up->weight == nullptr ||
        gate->weight->ne[0] != up->weight->ne[0] || gate->weight->ne[1] != up->weight->ne[1]) {
        return nullptr;
    }
    for (auto &candidate : context->swiglu_plans) {
        if (candidate.gate == gate && candidate.up == up) return &candidate;
    }

    fortai_cuda_q4_swiglu_plan candidate;
    candidate.gate = gate;
    candidate.up = up;
    candidate.backend = gate->backend;
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = true;
    candidate.context = ggml_init(params);
    if (candidate.context == nullptr) return nullptr;
    candidate.activation = ggml_new_tensor_2d(candidate.context, GGML_TYPE_F32,
        gate->weight->ne[0], 1);
    candidate.gate_output = ggml_mul_mat(candidate.context, gate->weight, candidate.activation);
    candidate.up_output = ggml_mul_mat(candidate.context, up->weight, candidate.activation);
    candidate.output = ggml_swiglu_split(candidate.context, candidate.gate_output,
        candidate.up_output);
    candidate.graph = ggml_new_graph_custom(candidate.context, GGML_DEFAULT_GRAPH_SIZE, false);
    if (candidate.activation == nullptr || candidate.gate_output == nullptr ||
        candidate.up_output == nullptr || candidate.output == nullptr || candidate.graph == nullptr) {
        ggml_free(candidate.context);
        return nullptr;
    }
    ggml_build_forward_expand(candidate.graph, candidate.output);
    candidate.compute_buffer = ggml_backend_alloc_ctx_tensors(candidate.context, candidate.backend);
    if (candidate.compute_buffer == nullptr) {
        ggml_free(candidate.context);
        return nullptr;
    }
    ggml_backend_buffer_set_usage(candidate.compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    try {
        context->swiglu_plans.push_back(candidate);
    } catch (...) {
        ggml_backend_buffer_free(candidate.compute_buffer);
        ggml_free(candidate.context);
        return nullptr;
    }
    return &context->swiglu_plans.back();
}

static fortai_cuda_q4_ffn_plan * fortai_cuda_q4_find_ffn_plan(
    fortai_cuda_q4_context *context, const fortai_cuda_q4_weights *gate,
    const fortai_cuda_q4_weights *up, const fortai_cuda_q4_weights *down) {
    if (context == nullptr || gate == nullptr || up == nullptr || down == nullptr ||
        gate->owner != context || up->owner != context || down->owner != context ||
        gate->backend != up->backend || gate->backend != down->backend ||
        gate->device != up->device || gate->device != down->device ||
        gate->weight == nullptr || up->weight == nullptr || down->weight == nullptr ||
        gate->weight->ne[0] != up->weight->ne[0] || gate->weight->ne[1] != up->weight->ne[1] ||
        down->weight->ne[0] != gate->weight->ne[1]) {
        return nullptr;
    }
    for (auto &candidate : context->ffn_plans) {
        if (candidate.gate == gate && candidate.up == up && candidate.down == down)
            return &candidate;
    }

    fortai_cuda_q4_ffn_plan candidate;
    candidate.gate = gate;
    candidate.up = up;
    candidate.down = down;
    candidate.backend = gate->backend;
    ggml_init_params params{};
    params.mem_size = 1 << 20;
    params.no_alloc = true;
    candidate.context = ggml_init(params);
    if (candidate.context == nullptr) return nullptr;
    candidate.activation = ggml_new_tensor_2d(candidate.context, GGML_TYPE_F32,
        gate->weight->ne[0], 1);
    candidate.gate_output = ggml_mul_mat(candidate.context, gate->weight, candidate.activation);
    candidate.up_output = ggml_mul_mat(candidate.context, up->weight, candidate.activation);
    candidate.swiglu_output = ggml_swiglu_split(candidate.context, candidate.gate_output,
        candidate.up_output);
    candidate.output = ggml_mul_mat(candidate.context, down->weight, candidate.swiglu_output);
    candidate.graph = ggml_new_graph_custom(candidate.context, GGML_DEFAULT_GRAPH_SIZE, false);
    if (candidate.activation == nullptr || candidate.gate_output == nullptr ||
        candidate.up_output == nullptr || candidate.swiglu_output == nullptr ||
        candidate.output == nullptr || candidate.graph == nullptr) {
        ggml_free(candidate.context);
        return nullptr;
    }
    ggml_build_forward_expand(candidate.graph, candidate.output);
    candidate.compute_buffer = ggml_backend_alloc_ctx_tensors(candidate.context, candidate.backend);
    if (candidate.compute_buffer == nullptr) {
        ggml_free(candidate.context);
        return nullptr;
    }
    ggml_backend_buffer_set_usage(candidate.compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    try {
        context->ffn_plans.push_back(candidate);
    } catch (...) {
        ggml_backend_buffer_free(candidate.compute_buffer);
        ggml_free(candidate.context);
        return nullptr;
    }
    return &context->ffn_plans.back();
}

static int fail(fortai_cuda_q4_context * context) {
    (void) context;
    return FORTAI_CUDA_RUNTIME_ERROR;
}

static void fortai_cuda_q4_debug(const char *operation) {
    const char *enabled = std::getenv("FORTAI_CUDA_Q4_DEBUG");
    if (enabled == nullptr || enabled[0] == '0') return;
    const cudaError_t error = cudaPeekAtLastError();
    std::fprintf(stderr, "fortai-q4: %s: %s (%d)\n", operation,
        cudaGetErrorString(error), static_cast<int>(error));
}

static void fortai_cuda_q4_debug_code(const char *operation, int code) {
    const char *enabled = std::getenv("FORTAI_CUDA_Q4_DEBUG");
    if (enabled == nullptr || enabled[0] == '0') return;
    const cudaError_t error = cudaPeekAtLastError();
    std::fprintf(stderr, "fortai-q4: %s: code=%d cuda=%s (%d)\n", operation,
        code, cudaGetErrorString(error), static_cast<int>(error));
}

static int fortai_cuda_q4_consumer_slot(const fortai_cuda_q4_context *context, int device) {
    if (context == nullptr) return -1;
    for (int slot = 0; slot < 2; ++slot)
        if (context->device_ids[slot] == device) return slot;
    return -1;
}

static cudaStream_t fortai_cuda_q4_scheduler_stream(
    fortai_cuda_q4_context *context, int slot) {
    if (context == nullptr || slot < 0 || slot > 1 ||
        context->devices[slot] == nullptr)
        return nullptr;
    auto *backend = reinterpret_cast<fortai_ggml_backend_runtime_layout *>(context->devices[slot]);
    if (backend == nullptr || backend->context == nullptr) return nullptr;
    auto *cuda_context = static_cast<ggml_backend_cuda_context *>(backend->context);
    cudaSetDevice(context->device_ids[slot]);
    return cuda_context->stream();
}

static int fortai_cuda_q4_record_scheduler_event(fortai_cuda_q4_context *context,
    int slot, ggml_backend_event_t event) {
    if (context == nullptr || event == nullptr || slot < 0 || slot > 1)
        return FORTAI_CUDA_INVALID;
    const cudaStream_t stream = fortai_cuda_q4_scheduler_stream(context, slot);
    if (stream == nullptr || cudaEventRecord(fortai_cuda_event_handle(event), stream) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    return FORTAI_CUDA_OK;
}

static int fortai_cuda_q4_wait_scheduler_event(fortai_cuda_q4_context *context,
    int slot, ggml_backend_event_t event) {
    if (context == nullptr || event == nullptr || slot < 0 || slot > 1)
        return FORTAI_CUDA_INVALID;
    const cudaStream_t stream = fortai_cuda_q4_scheduler_stream(context, slot);
    if (stream == nullptr || cudaStreamWaitEvent(stream,
            fortai_cuda_event_handle(event), 0) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    return FORTAI_CUDA_OK;
}

static bool fortai_cuda_q4_sync_bridge_requested() {
    const char *value = std::getenv("FORTAI_CUDA_Q4_SYNC_BRIDGE");
    return value != nullptr && (value[0] == '1' || value[0] == 'y' ||
        value[0] == 'Y' || value[0] == 't' || value[0] == 'T');
}

static int fortai_cuda_q4_next_event(fortai_cuda_q4_context *context, int slot,
    ggml_backend_event_t events[][fortai_cuda_q4_context::event_slots], unsigned indices[2],
    ggml_backend_event_t *event) {
    if (context == nullptr || slot < 0 || slot > 1 || event == nullptr) return FORTAI_CUDA_INVALID;
    if (cudaSetDevice(context->device_ids[slot]) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    const unsigned sequence = indices[slot]++;
    const unsigned index = sequence % fortai_cuda_q4_context::event_slots;
    *event = events[slot][index];
    if (*event == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
    if (sequence >= fortai_cuda_q4_context::event_slots) {
        const cudaError_t query = cudaEventQuery(fortai_cuda_event_handle(*event));
        if (query == cudaErrorNotReady) {
            if (cudaEventSynchronize(fortai_cuda_event_handle(*event)) != cudaSuccess)
                return FORTAI_CUDA_RUNTIME_ERROR;
        } else if (query != cudaSuccess) {
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
    }
    return FORTAI_CUDA_OK;
}

/* GGML may use several internal streams for a graph.  The attached FortAI
 * stream is the producer of the activation consumed by Q4.  Queue a backend
 * event wait rather than synchronizing the host; a device-wide synchronize
 * would also stall unrelated work on the second board. */
static int fortai_cuda_q4_prepare_input(fortai_cuda_q4_context *context, int device) {
    const int slot = fortai_cuda_q4_consumer_slot(context, device);
    if (slot < 0) return FORTAI_CUDA_INVALID;
    if (context->consumer_stream[slot] == nullptr) return FORTAI_CUDA_OK;
    if (context->consumer_stream[slot] == fortai_cuda_q4_scheduler_stream(context, slot))
        return FORTAI_CUDA_OK;
    ggml_backend_event_t producer_event = nullptr;
    if (fortai_cuda_q4_next_event(context, slot, context->producer_event,
            context->producer_event_index, &producer_event) != FORTAI_CUDA_OK)
        return FORTAI_CUDA_RUNTIME_ERROR;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaEventRecord(fortai_cuda_event_handle(producer_event), context->consumer_stream[slot]) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    /* Queue a device-side dependency on GGML's own stream.  The old path
     * synchronized this event on the host, which serialized every Q4
     * projection behind the preceding Q8 operation. */
    return fortai_cuda_q4_wait_scheduler_event(context, slot, producer_event);
}

static int fortai_cuda_q4_publish_output(fortai_cuda_q4_context *context,
    ggml_backend_t backend, int device) {
    const int slot = fortai_cuda_q4_consumer_slot(context, device);
    if (slot < 0) return FORTAI_CUDA_INVALID;
    if (context->consumer_stream[slot] == nullptr) return FORTAI_CUDA_OK;
    if (context->consumer_stream[slot] == fortai_cuda_q4_scheduler_stream(context, slot))
        return FORTAI_CUDA_OK;
    ggml_backend_event_t output_event = nullptr;
    if (fortai_cuda_q4_next_event(context, slot, context->producer_event,
            context->producer_event_index, &output_event) != FORTAI_CUDA_OK)
        return FORTAI_CUDA_RUNTIME_ERROR;
    /* The graph and the event use the same GGML stream.  Once recorded, the
     * downstream Fortran stream can wait without a host-visible scheduler
     * barrier, preserving the dependency for the next native CUDA op. */
    if (fortai_cuda_q4_record_scheduler_event(context, slot, output_event) != FORTAI_CUDA_OK ||
        cudaSetDevice(device) != cudaSuccess ||
        cudaStreamWaitEvent(context->consumer_stream[slot],
            fortai_cuda_event_handle(output_event), 0) != cudaSuccess) {
        fortai_cuda_q4_debug("publish output edge");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
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
    created->segment_graph_requested = fortai_cuda_q4_segment_graph_requested();
    created->sync_bridge_requested = fortai_cuda_q4_sync_bridge_requested();
    /* The unified and split-segment paths capture native and Q4 operations
     * in an outer graph.  GGML's per-plan graph must therefore execute
     * directly; nesting its capture in the active outer capture is rejected
     * by CUDA.  Segment capture is safe for split devices because it stops at
     * the first remote layer, before the scheduler's bridge streams. */
    bool disable_ggml_graphs = false;
    if (const char *unified = std::getenv("FORTAI_ENABLE_CUDA_Q4_UNIFIED_GRAPH")) {
        disable_ggml_graphs = unified[0] == '1' || unified[0] == 'y' || unified[0] == 'Y' ||
            unified[0] == 't' || unified[0] == 'T';
        if (disable_ggml_graphs && created->device_ids[0] != created->device_ids[1])
            disable_ggml_graphs = false;
    }
    if (const char *segment = std::getenv("FORTAI_ENABLE_CUDA_Q4_SEGMENT_GRAPH")) {
        if (segment[0] == '1' || segment[0] == 'y' || segment[0] == 'Y' ||
            segment[0] == 't' || segment[0] == 'T')
            disable_ggml_graphs = true;
    }
    if (disable_ggml_graphs)
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);
    if (const char *peer_copy = std::getenv("FORTAI_CUDA_Q4_USE_PEER")) {
        if (peer_copy[0] == '1' || peer_copy[0] == 'y' || peer_copy[0] == 'Y' ||
            peer_copy[0] == 't' || peer_copy[0] == 'T')
            created->use_peer_copy = true;
    }
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
        cudaSetDevice(first_device);
        if (cudaStreamCreateWithFlags(&created->bridge_stream[0], cudaStreamNonBlocking) != cudaSuccess) {
            fortai_cuda_q4_context_destroy(created);
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        cudaSetDevice(second_device);
        if (cudaStreamCreateWithFlags(&created->bridge_stream[1], cudaStreamNonBlocking) != cudaSuccess) {
            fortai_cuda_q4_context_destroy(created);
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        if (cudaMalloc(&created->bridge_activation[1], q4_bridge_activation_reserve) != cudaSuccess ||
            cudaMalloc(&created->bridge_output[1], q4_bridge_output_reserve) != cudaSuccess) {
            fortai_cuda_q4_context_destroy(created);
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        created->bridge_activation_bytes[1] = q4_bridge_activation_reserve;
        created->bridge_output_bytes[1] = q4_bridge_output_reserve;

        /* These raw events protect the pinned host rings.  They are created
         * on the device whose stream records them, then waited from the
         * opposite bridge stream; CUDA supports this cross-device event edge
         * even on PCIe-only systems without peer memory access. */
        constexpr unsigned event_flags = cudaEventDisableTiming;
        cudaSetDevice(first_device);
        for (int i = 0; i < fortai_cuda_q4_context::host_bridge_slots; ++i) {
            if (cudaEventCreateWithFlags(&created->input_host[i].ready_event, event_flags) != cudaSuccess ||
                cudaEventCreateWithFlags(&created->output_host[i].free_event, event_flags) != cudaSuccess) {
                fortai_cuda_q4_context_destroy(created);
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        }
        cudaSetDevice(second_device);
        for (int i = 0; i < fortai_cuda_q4_context::host_bridge_slots; ++i) {
            if (cudaEventCreateWithFlags(&created->input_host[i].free_event, event_flags) != cudaSuccess) {
                fortai_cuda_q4_context_destroy(created);
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        }

        cudaSetDevice(first_device);
        for (int event = 0; event < fortai_cuda_q4_context::event_slots; ++event) {
            if (fortai_cuda_q4_create_bridge_events(created, &created->bridge_events[event]) !=
                    FORTAI_CUDA_OK) {
                fortai_cuda_q4_context_destroy(created);
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        }
    }
    for (int slot = 0; slot < 2; ++slot) {
        cudaSetDevice(created->device_ids[slot]);
        for (int event = 0; event < fortai_cuda_q4_context::event_slots; ++event) {
            created->producer_event[slot][event] = ggml_backend_event_new(
                ggml_backend_get_device(created->devices[slot]));
            if (created->producer_event[slot][event] == nullptr) {
                fortai_cuda_q4_context_destroy(created);
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        }
    }
    *out = created;
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_context_destroy(fortai_cuda_q4_context * context) {
    if (context == nullptr) return FORTAI_CUDA_OK;
    for (int i = 0; i < 2; ++i) {
        if (context->bridge_stream[i] != nullptr) {
            cudaSetDevice(context->device_ids[i]);
            cudaStreamSynchronize(context->bridge_stream[i]);
        }
    }
    for (int i = 0; i < 2; ++i) {
        cudaSetDevice(context->device_ids[i]);
        if (context->bridge_stream[i] != nullptr) {
            cudaStreamDestroy(context->bridge_stream[i]);
            context->bridge_stream[i] = nullptr;
        }
        if (i == 0 && context->device_ids[0] != context->device_ids[1]) {
            for (int event = 0; event < fortai_cuda_q4_context::event_slots; ++event) {
                if (context->bridge_events[event].source_ready != nullptr)
                    ggml_backend_event_free(context->bridge_events[event].source_ready);
                if (context->bridge_events[event].primary_done != nullptr)
                    ggml_backend_event_free(context->bridge_events[event].primary_done);
                context->bridge_events[event].source_ready = nullptr;
                context->bridge_events[event].primary_done = nullptr;
            }
        }
        if (i == 1 && context->device_ids[0] != context->device_ids[1]) {
            for (int event = 0; event < fortai_cuda_q4_context::event_slots; ++event) {
                if (context->bridge_events[event].remote_ready != nullptr)
                    ggml_backend_event_free(context->bridge_events[event].remote_ready);
                if (context->bridge_events[event].remote_done != nullptr)
                    ggml_backend_event_free(context->bridge_events[event].remote_done);
                if (context->bridge_events[event].output_free != nullptr)
                    ggml_backend_event_free(context->bridge_events[event].output_free);
                context->bridge_events[event].remote_ready = nullptr;
                context->bridge_events[event].remote_done = nullptr;
                context->bridge_events[event].output_free = nullptr;
            }
        }
        for (int event = 0; event < fortai_cuda_q4_context::event_slots; ++event) {
            if (context->producer_event[i][event] != nullptr)
                ggml_backend_event_free(context->producer_event[i][event]);
        }
        if (context->input_host[i].free_event != nullptr) {
            cudaEventDestroy(context->input_host[i].free_event);
            context->input_host[i].free_event = nullptr;
        }
        if (context->input_host[i].ready_event != nullptr) {
            cudaEventDestroy(context->input_host[i].ready_event);
            context->input_host[i].ready_event = nullptr;
        }
        if (context->output_host[i].free_event != nullptr) {
            cudaEventDestroy(context->output_host[i].free_event);
            context->output_host[i].free_event = nullptr;
        }
        cudaFreeHost(context->input_host[i].buffer);
        cudaFreeHost(context->output_host[i].buffer);
        context->input_host[i].buffer = nullptr;
        context->output_host[i].buffer = nullptr;
        cudaFree(context->bridge_activation[i]);
        cudaFree(context->bridge_output[i]);
    }
    for (auto *events : context->extra_bridge_events) {
        fortai_cuda_q4_free_bridge_events(context, events);
        delete events;
    }
    context->extra_bridge_events.clear();
    auto destroy_cached_graph = [](cudaGraph_t &graph, cudaGraphExec_t &exec) {
        if (exec != nullptr) cudaGraphExecDestroy(exec);
        if (graph != nullptr) cudaGraphDestroy(graph);
        exec = nullptr;
        graph = nullptr;
    };
    for (auto &plan : context->batch_plans) {
        destroy_cached_graph(plan.cached_graph, plan.cached_exec);
        if (plan.buffer != nullptr) ggml_backend_buffer_free(plan.buffer);
        plan.buffer = nullptr;
        if (plan.context != nullptr) ggml_free(plan.context);
        plan.context = nullptr;
    }
    context->batch_plans.clear();
    for (auto &plan : context->embedding_batch_plans) {
        destroy_cached_graph(plan.cached_graph, plan.cached_exec);
        if (plan.buffer != nullptr) ggml_backend_buffer_free(plan.buffer);
        plan.buffer = nullptr;
        if (plan.context != nullptr) ggml_free(plan.context);
        plan.context = nullptr;
    }
    context->embedding_batch_plans.clear();
    for (auto &plan : context->group_plans)
        destroy_cached_graph(plan.cached_graph, plan.cached_exec);
    context->group_plans.clear();
    for (auto &plan : context->swiglu_plans) {
        destroy_cached_graph(plan.cached_graph, plan.cached_exec);
        if (plan.compute_buffer != nullptr) ggml_backend_buffer_free(plan.compute_buffer);
        plan.compute_buffer = nullptr;
        if (plan.context != nullptr) ggml_free(plan.context);
        plan.context = nullptr;
    }
    context->swiglu_plans.clear();
    for (auto &plan : context->ffn_plans) {
        destroy_cached_graph(plan.cached_graph, plan.cached_exec);
        if (plan.compute_buffer != nullptr) ggml_backend_buffer_free(plan.compute_buffer);
        plan.compute_buffer = nullptr;
        if (plan.context != nullptr) ggml_free(plan.context);
        plan.context = nullptr;
    }
    context->ffn_plans.clear();
    for (int slot = 0; slot < 2; ++slot) {
        for (auto &arena : context->weight_arenas[slot]) {
            if (arena.buffer != nullptr) ggml_backend_buffer_free(arena.buffer);
            arena.buffer = nullptr;
            arena.used = 0;
            arena.capacity = 0;
        }
        context->weight_arenas[slot].clear();
    }
    cudaFreeHost(context->host_bridge);
    if (context->devices[0] != nullptr) ggml_backend_free(context->devices[0]);
    if (context->devices[1] != nullptr && context->devices[1] != context->devices[0])
        ggml_backend_free(context->devices[1]);
    if (context->cpu != nullptr) ggml_backend_free(context->cpu);
    delete context;
    fortai_quiet_log_release();
    return FORTAI_CUDA_OK;
}

/* A separate GGML context per projection used to imply one cudaMalloc per
 * tensor.  UD-Q4_K_XL has hundreds of such tensors, so allocation and upload
 * overhead dominated model open.  Keep the graph contexts independent but
 * suballocate their immutable weights from a small number of device arenas,
 * matching llama.cpp's packed model buffers without changing the ABI. */
static int fortai_cuda_q4_arena_alloc(fortai_cuda_q4_context *context, int device_slot,
    size_t bytes, ggml_backend_buffer_t *buffer, void **address) {
    if (context == nullptr || buffer == nullptr || address == nullptr || device_slot < 0 ||
        device_slot > 1 || bytes == 0)
        return FORTAI_CUDA_INVALID;
    const ggml_backend_t backend = context->devices[device_slot];
    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = std::max<size_t>(1, ggml_backend_buft_get_alignment(buft));
    /* A half-gigabyte chunk keeps the allocation count low enough for CUDA's
     * VMM allocator while bounding tail slack below the long-context headroom
     * on the 16 GiB boards. */
    constexpr size_t arena_chunk_bytes = size_t(512) << 20;
    auto &arenas = context->weight_arenas[device_slot];
    fortai_cuda_q4_weight_arena *arena = arenas.empty() ? nullptr : &arenas.back();
    size_t offset = 0;
    if (arena != nullptr) {
        offset = (arena->used + alignment - 1) / alignment * alignment;
        if (offset > arena->capacity || bytes > arena->capacity - offset) arena = nullptr;
    }
    if (arena == nullptr) {
        const size_t capacity = std::max(arena_chunk_bytes,
            (bytes + alignment - 1) / alignment * alignment);
        ggml_backend_buffer_t storage = ggml_backend_buft_alloc_buffer(buft, capacity);
        if (storage == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
        try {
            arenas.push_back({storage, 0, capacity});
        } catch (...) {
            ggml_backend_buffer_free(storage);
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        arena = &arenas.back();
        offset = 0;
    }
    *buffer = arena->buffer;
    *address = static_cast<char *>(ggml_backend_buffer_get_base(arena->buffer)) + offset;
    arena->used = offset + bytes;
    return FORTAI_CUDA_OK;
}

/* CUDA treats a pageable source for cudaMemcpyAsync as a synchronous staging
 * operation.  UD-Q4_K_XL has hundreds of mapped tensors, so issuing one such
 * copy per tensor makes model loading dominated by driver staging overhead.
 * Reuse one bounded pinned buffer and perform blocking copies during setup;
 * the upload is outside the decode critical path and reaches the same DMA
 * engine used by llama.cpp's pinned loader without retaining the whole model
 * in locked host memory. */
static bool fortai_cuda_q4_staged_upload_enabled() {
    const char *value = std::getenv("FORTAI_CUDA_STAGED_UPLOAD");
    return value == nullptr || (value[0] != '0' && value[0] != 'n' && value[0] != 'N' &&
        value[0] != 'f' && value[0] != 'F');
}

static int fortai_cuda_q4_upload_staged(fortai_cuda_q4_context *context, int device,
    void *destination, const void *source, size_t bytes) {
    if (context == nullptr || destination == nullptr || source == nullptr || bytes == 0)
        return FORTAI_CUDA_INVALID;
    constexpr size_t stage_capacity = size_t(64) << 20;
    const size_t required = std::min(stage_capacity, bytes);
    if (context->host_bridge_bytes < required) {
        if (context->host_bridge != nullptr) {
            cudaFreeHost(context->host_bridge);
            context->host_bridge = nullptr;
            context->host_bridge_bytes = 0;
        }
        if (cudaHostAlloc(&context->host_bridge, required, cudaHostAllocPortable) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        context->host_bridge_bytes = required;
    }
    if (cudaSetDevice(device) != cudaSuccess) return FORTAI_CUDA_RUNTIME_ERROR;
    const auto *input = static_cast<const unsigned char *>(source);
    auto *output = static_cast<unsigned char *>(destination);
    for (size_t offset = 0; offset < bytes; offset += required) {
        const size_t chunk = std::min(required, bytes - offset);
        std::memcpy(context->host_bridge, input + offset, chunk);
        if (cudaMemcpy(output + offset, context->host_bridge, chunk,
                cudaMemcpyHostToDevice) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
    }
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
    const int device_slot = fortai_cuda_q4_consumer_slot(context, created->device);
    if (device_slot < 0) {
        fortai_cuda_q4_weights_destroy(created);
        return FORTAI_CUDA_INVALID;
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(created->backend);
    const size_t weight_alloc_bytes = ggml_backend_buft_get_alloc_size(buft, created->weight);
    void *weight_address = nullptr;
    if (fortai_cuda_q4_arena_alloc(context, device_slot, weight_alloc_bytes,
            &created->buffer, &weight_address) != FORTAI_CUDA_OK ||
        ggml_backend_tensor_alloc(created->buffer, created->weight, weight_address) != GGML_STATUS_SUCCESS) {
        created->buffer = nullptr;
        fortai_cuda_q4_weights_destroy(created);
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    ggml_build_forward_expand(created->graph, created->output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_alloc_ctx_tensors(created->graph_context, created->backend);
    if (compute_buffer == nullptr) {
        created->buffer = nullptr; // the arena owns the weight storage
        fortai_cuda_q4_weights_destroy(created);
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    created->buffer = compute_buffer;
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    /* Mapped GGUF pages are pageable.  Stage through one bounded pinned
     * buffer when possible; fall back to GGML's async path if the host cannot
     * provide a pinned allocation (for example under a strict memlock limit).
     */
    const int staged = fortai_cuda_q4_staged_upload_enabled() ?
        fortai_cuda_q4_upload_staged(context, created->device, weight_address, host_weights, weight_bytes) :
        FORTAI_CUDA_RUNTIME_ERROR;
    if (staged != FORTAI_CUDA_OK)
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
    for (int slot = 0; slot < 2; ++slot) {
        if (context->bridge_stream[slot] != nullptr) {
            if (cudaSetDevice(context->device_ids[slot]) != cudaSuccess ||
                cudaStreamSynchronize(context->bridge_stream[slot]) != cudaSuccess)
                return FORTAI_CUDA_RUNTIME_ERROR;
        }
    }
    /* The host bridge doubles as the bounded setup staging buffer.  Release
     * it once all model uploads are complete; runtime transfers allocate it
     * lazily again only if a non-peer bridge is actually needed. */
    cudaFreeHost(context->host_bridge);
    context->host_bridge = nullptr;
    context->host_bridge_bytes = 0;
    return FORTAI_CUDA_OK;
}

void *fortai_cuda_q4_context_stream(fortai_cuda_q4_context *context,
    int device_slot) {
    if (context == nullptr || device_slot < 0 || device_slot > 1 ||
        context->devices[device_slot] == nullptr)
        return nullptr;
    auto *backend = reinterpret_cast<fortai_ggml_backend_runtime_layout *>(context->devices[device_slot]);
    if (backend == nullptr || backend->context == nullptr) return nullptr;
    auto *cuda_context = static_cast<ggml_backend_cuda_context *>(backend->context);
    cudaSetDevice(context->device_ids[device_slot]);
    return reinterpret_cast<void *>(cuda_context->stream());
}

static bool fortai_cuda_q4_segment_graph_requested() {
    const char *value = std::getenv("FORTAI_ENABLE_CUDA_Q4_SEGMENT_GRAPH");
    return value != nullptr && (value[0] == '1' || value[0] == 'y' ||
        value[0] == 'Y' || value[0] == 't' || value[0] == 'T');
}

static int fortai_cuda_q4_backend_slot(const fortai_cuda_q4_context *context,
    ggml_backend_t backend) {
    if (context == nullptr || backend == nullptr) return -1;
    for (int slot = 0; slot < 2; ++slot)
        if (context->devices[slot] == backend) return slot;
    return -1;
}

/* GGML-CUDA normally captures one graph per backend scheduler.  The split
 * Q4 path also has a native CUDA graph for the primary prefix, so GGML's
 * process-wide graph switch is disabled there to avoid nested capture.  Keep
 * the same low-overhead behaviour for every Q4 plan outside that prefix by
 * caching a small graph per stable plan.  The bridge waits are queued before
 * this helper and therefore remain outside the cached graph, just as they do
 * in llama.cpp's scheduler split. */
static enum ggml_status fortai_cuda_q4_graph_compute_cached(
    fortai_cuda_q4_context *context, ggml_backend_t backend, ggml_cgraph *graph,
    cudaGraph_t *cached_graph, cudaGraphExec_t *cached_exec, bool *warmup,
    bool *disabled) {
    if (context == nullptr || backend == nullptr || graph == nullptr ||
        cached_graph == nullptr || cached_exec == nullptr || warmup == nullptr ||
        disabled == nullptr)
        return GGML_STATUS_FAILED;
    if (!context->segment_graph_requested || *disabled)
        return ggml_backend_graph_compute_async(backend, graph);

    const int slot = fortai_cuda_q4_backend_slot(context, backend);
    const cudaStream_t stream = fortai_cuda_q4_scheduler_stream(context, slot);
    if (slot < 0 || stream == nullptr)
        return ggml_backend_graph_compute_async(backend, graph);

    enum cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture_status) != cudaSuccess)
        return ggml_backend_graph_compute_async(backend, graph);
    /* The native prefix graph owns this capture.  Directly enqueueing the
     * kernels is exactly what becomes part of that outer graph. */
    if (capture_status == cudaStreamCaptureStatusActive)
        return ggml_backend_graph_compute_async(backend, graph);

    if (*cached_exec != nullptr) {
        if (cudaSetDevice(context->device_ids[slot]) != cudaSuccess ||
            cudaGraphLaunch(*cached_exec, stream) != cudaSuccess)
            return GGML_STATUS_FAILED;
        return GGML_STATUS_SUCCESS;
    }

    if (!*warmup) {
        *warmup = true;
        return ggml_backend_graph_compute_async(backend, graph);
    }

    if (cudaSetDevice(context->device_ids[slot]) != cudaSuccess ||
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
        *disabled = true;
        return ggml_backend_graph_compute_async(backend, graph);
    }
    const enum ggml_status status = ggml_backend_graph_compute_async(backend, graph);
    cudaGraph_t captured = nullptr;
    const cudaError_t end_error = cudaStreamEndCapture(stream, &captured);
    if (status != GGML_STATUS_SUCCESS || end_error != cudaSuccess || captured == nullptr) {
        if (captured != nullptr) cudaGraphDestroy(captured);
        *disabled = true;
        return ggml_backend_graph_compute_async(backend, graph);
    }
    cudaGraphExec_t executable = nullptr;
    const cudaError_t instantiate_error = cudaGraphInstantiate(&executable, captured,
        nullptr, nullptr, 0);
    if (instantiate_error != cudaSuccess || executable == nullptr) {
        cudaGraphDestroy(captured);
        *disabled = true;
        return ggml_backend_graph_compute_async(backend, graph);
    }
    *cached_graph = captured;
    *cached_exec = executable;
    if (cudaGraphLaunch(*cached_exec, stream) != cudaSuccess)
        return GGML_STATUS_FAILED;
    return GGML_STATUS_SUCCESS;
}

int fortai_cuda_q4_context_set_consumer_stream(fortai_cuda_q4_context *context,
    int device_slot, void *stream) {
    if (context == nullptr || device_slot < 0 || device_slot > 1)
        return FORTAI_CUDA_INVALID;
    context->consumer_stream[device_slot] = reinterpret_cast<cudaStream_t>(stream);
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_weights_destroy(fortai_cuda_q4_weights * weights) {
    if (weights == nullptr) return FORTAI_CUDA_OK;
    if (weights->cached_exec != nullptr) cudaGraphExecDestroy(weights->cached_exec);
    if (weights->cached_graph != nullptr) cudaGraphDestroy(weights->cached_graph);
    if (weights->embedding_cached_exec != nullptr) cudaGraphExecDestroy(weights->embedding_cached_exec);
    if (weights->embedding_cached_graph != nullptr) cudaGraphDestroy(weights->embedding_cached_graph);
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
            enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context,
                weight->backend, weight->graph,
                &const_cast<fortai_cuda_q4_weights *>(weight)->cached_graph,
                &const_cast<fortai_cuda_q4_weights *>(weight)->cached_exec,
                &const_cast<fortai_cuda_q4_weights *>(weight)->graph_warmup,
                &const_cast<fortai_cuda_q4_weights *>(weight)->graph_disabled);
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
        enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context, plan->backend, plan->graph,
            &plan->cached_graph, &plan->cached_exec, &plan->graph_warmup, &plan->graph_disabled);
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

/* The two RTX 5060 Ti boards in the target host do not expose peer access.
 * cudaMemcpyPeer still reports success on that topology, but its implicit
 * staging path is not reliable for every mixed-Q4 allocation.  Make the
 * fallback explicit so a remote projection always sees the exact activation
 * and its result is copied back without relying on peer/UVA state. */
static int fortai_cuda_q4_copy_between_devices(fortai_cuda_q4_context *context,
    void *destination, int destination_device, const void *source, int source_device, size_t bytes) {
    if (context == nullptr) return FORTAI_CUDA_INVALID;
    if (destination == nullptr || source == nullptr || bytes == 0)
        return FORTAI_CUDA_INVALID;
    if (destination_device == source_device) {
        if (cudaSetDevice(destination_device) != cudaSuccess ||
            cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        return FORTAI_CUDA_OK;
    }
    if (context->use_peer_copy) {
        if (cudaSetDevice(destination_device) != cudaSuccess ||
            cudaMemcpyPeer(destination, destination_device, source, source_device, bytes) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        return FORTAI_CUDA_OK;
    }
    /* The diagnostic synchronous bridge is deliberately conservative: a
     * plain cudaMemcpy does not establish a dependency on GGML's non-default
     * producer stream on every runtime configuration.  Drain that source
     * stream before the host staging copy so this A/B tests bridge data, not
     * an unrelated stream race. */
    if (context->sync_bridge_requested) {
        if (cudaSetDevice(source_device) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
    }
    if (context->host_bridge_bytes < bytes) {
        cudaFreeHost(context->host_bridge);
        context->host_bridge = nullptr;
        if (cudaHostAlloc(&context->host_bridge, bytes, cudaHostAllocPortable) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        context->host_bridge_bytes = bytes;
    }
    if (cudaSetDevice(source_device) != cudaSuccess ||
        cudaMemcpy(context->host_bridge, source, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    if (cudaSetDevice(destination_device) != cudaSuccess ||
        cudaMemcpy(destination, context->host_bridge, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    /* A synchronous host copy completes from the caller's point of view, but
     * it is still issued on the current/default stream.  The native consumer
     * runs on a non-default stream, so make the diagnostic path publish an
     * explicit device-wide dependency before returning to Fortran. */
    if (context->sync_bridge_requested && cudaDeviceSynchronize() != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    return FORTAI_CUDA_OK;
}

/* Queue a primary-device activation through the two pinned rings.  The
 * source event is recorded on FortAI's producer stream, the D2H/H2D copies
 * run on the bridge streams, and GGML's private stream waits for remote_ready.
 * No host synchronization is needed on the normal resident-device path. */
static int fortai_cuda_q4_queue_activation_remote(fortai_cuda_q4_context *context,
    const void *device_activation, size_t activation_bytes,
    fortai_cuda_q4_bridge_events **events_out) {
    if (context == nullptr || device_activation == nullptr || activation_bytes == 0 ||
        context->bridge_stream[0] == nullptr || context->bridge_stream[1] == nullptr ||
        context->consumer_stream[0] == nullptr || events_out == nullptr)
        return FORTAI_CUDA_INVALID;
    fortai_cuda_q4_bridge_events *events = nullptr;
    if (fortai_cuda_q4_next_bridge_events(context, &events) != FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug("next bridge events");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    const int primary_device = context->device_ids[0];
    const int remote_device = context->device_ids[1];

    if (context->use_peer_copy) {
        if (cudaSetDevice(primary_device) != cudaSuccess ||
            cudaEventRecord(fortai_cuda_event_handle(events->source_ready),
                context->consumer_stream[0]) != cudaSuccess) {
            fortai_cuda_q4_debug("peer activation source");
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        if (cudaSetDevice(remote_device) != cudaSuccess ||
            (context->last_activation_done != nullptr &&
                cudaStreamWaitEvent(context->bridge_stream[1],
                    fortai_cuda_event_handle(context->last_activation_done), 0) != cudaSuccess) ||
            cudaStreamWaitEvent(context->bridge_stream[1],
                fortai_cuda_event_handle(events->source_ready), 0) != cudaSuccess ||
            cudaMemcpyPeerAsync(context->bridge_activation[1], remote_device,
                device_activation, primary_device, activation_bytes,
                context->bridge_stream[1]) != cudaSuccess ||
            cudaEventRecord(fortai_cuda_event_handle(events->remote_ready),
                context->bridge_stream[1]) != cudaSuccess) {
            fortai_cuda_q4_debug("peer activation copy");
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        if (fortai_cuda_q4_wait_scheduler_event(context, 1, events->remote_ready) != FORTAI_CUDA_OK) {
            fortai_cuda_q4_debug("peer activation scheduler wait");
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        *events_out = events;
        return FORTAI_CUDA_OK;
    }

    int host_index = -1;
    if (fortai_cuda_q4_acquire_input_host(context, activation_bytes, &host_index) != FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug("acquire input bridge");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    auto &host = context->input_host[host_index];

    if (cudaSetDevice(primary_device) != cudaSuccess ||
        cudaEventRecord(fortai_cuda_event_handle(events->source_ready),
            context->consumer_stream[0]) != cudaSuccess ||
        cudaStreamWaitEvent(context->bridge_stream[0],
            fortai_cuda_event_handle(events->source_ready), 0) != cudaSuccess ||
        cudaMemcpyAsync(host.buffer, device_activation, activation_bytes,
            cudaMemcpyDeviceToHost, context->bridge_stream[0]) != cudaSuccess ||
        cudaEventRecord(host.ready_event, context->bridge_stream[0]) != cudaSuccess) {
        fortai_cuda_q4_debug("queue input bridge primary");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }

    if (cudaSetDevice(remote_device) != cudaSuccess ||
        context->last_activation_done != nullptr &&
            cudaStreamWaitEvent(context->bridge_stream[1],
                fortai_cuda_event_handle(context->last_activation_done), 0) != cudaSuccess ||
        cudaStreamWaitEvent(context->bridge_stream[1], host.ready_event, 0) != cudaSuccess ||
        cudaMemcpyAsync(context->bridge_activation[1], host.buffer, activation_bytes,
            cudaMemcpyHostToDevice, context->bridge_stream[1]) != cudaSuccess ||
        cudaEventRecord(host.free_event, context->bridge_stream[1]) != cudaSuccess ||
        cudaEventRecord(fortai_cuda_event_handle(events->remote_ready),
            context->bridge_stream[1]) != cudaSuccess) {
        fortai_cuda_q4_debug("queue input bridge remote");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    host.free_recorded = true;
    if (fortai_cuda_q4_wait_scheduler_event(context, 1, events->remote_ready) != FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug("input scheduler wait");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    *events_out = events;
    return FORTAI_CUDA_OK;
}

static int fortai_cuda_q4_record_remote_done(fortai_cuda_q4_context *context,
    ggml_backend_t backend, fortai_cuda_q4_bridge_events *events) {
    if (context == nullptr || backend == nullptr || events == nullptr) return FORTAI_CUDA_INVALID;
    const int slot = fortai_cuda_q4_consumer_slot(context, context->device_ids[1]);
    if (fortai_cuda_q4_record_scheduler_event(context, slot, events->remote_done) != FORTAI_CUDA_OK)
        return FORTAI_CUDA_RUNTIME_ERROR;
    context->last_activation_done = events->remote_done;
    return FORTAI_CUDA_OK;
}

/* Queue one or more remote outputs back to primary resident buffers.  The
 * remote graph records remote_done, then one D2H sequence fills pinned output
 * slots.  A single output_free edge protects the fixed GGML bridge buffer;
 * the primary bridge stream performs all H2D copies and publishes one event
 * to the downstream Q8 stream. */
static int fortai_cuda_q4_queue_outputs_primary(fortai_cuda_q4_context *context,
    fortai_cuda_q4_bridge_events *events, const void * const *sources,
    void * const *destinations, const size_t *bytes, int count) {
    if (context == nullptr || events == nullptr || sources == nullptr || destinations == nullptr ||
        bytes == nullptr || count < 1 || count > 8 || context->bridge_stream[0] == nullptr ||
        context->bridge_stream[1] == nullptr || context->consumer_stream[0] == nullptr)
        return FORTAI_CUDA_INVALID;
    int host_indices[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    const int primary_device = context->device_ids[0];
    const int remote_device = context->device_ids[1];
    if (context->use_peer_copy) {
        if (cudaSetDevice(primary_device) != cudaSuccess ||
            cudaStreamWaitEvent(context->bridge_stream[0],
                fortai_cuda_event_handle(events->remote_done), 0) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        for (int i = 0; i < count; ++i) {
            if (sources[i] == nullptr || destinations[i] == nullptr || bytes[i] == 0 ||
                cudaMemcpyPeerAsync(destinations[i], primary_device, sources[i], remote_device,
                    bytes[i], context->bridge_stream[0]) != cudaSuccess)
                return FORTAI_CUDA_RUNTIME_ERROR;
        }
        if (cudaEventRecord(fortai_cuda_event_handle(events->primary_done),
                context->bridge_stream[0]) != cudaSuccess ||
            cudaStreamWaitEvent(context->consumer_stream[0],
                fortai_cuda_event_handle(events->primary_done), 0) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        return FORTAI_CUDA_OK;
    }
    for (int i = 0; i < count; ++i) {
        if (sources[i] == nullptr || destinations[i] == nullptr || bytes[i] == 0 ||
            fortai_cuda_q4_acquire_output_host(context, bytes[i], &host_indices[i]) != FORTAI_CUDA_OK)
            return FORTAI_CUDA_RUNTIME_ERROR;
    }
    if (cudaSetDevice(remote_device) != cudaSuccess ||
        cudaStreamWaitEvent(context->bridge_stream[1],
            fortai_cuda_event_handle(events->remote_done), 0) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    for (int i = 0; i < count; ++i) {
        auto &host = context->output_host[host_indices[i]];
        if (cudaMemcpyAsync(host.buffer, sources[i], bytes[i], cudaMemcpyDeviceToHost,
                context->bridge_stream[1]) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
    }
    if (cudaEventRecord(fortai_cuda_event_handle(events->output_free),
            context->bridge_stream[1]) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    context->last_output_free = events->output_free;

    if (cudaSetDevice(primary_device) != cudaSuccess ||
        cudaStreamWaitEvent(context->bridge_stream[0],
            fortai_cuda_event_handle(events->output_free), 0) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    for (int i = 0; i < count; ++i) {
        auto &host = context->output_host[host_indices[i]];
        if (cudaMemcpyAsync(destinations[i], host.buffer, bytes[i], cudaMemcpyHostToDevice,
                context->bridge_stream[0]) != cudaSuccess ||
            cudaEventRecord(host.free_event, context->bridge_stream[0]) != cudaSuccess)
            return FORTAI_CUDA_RUNTIME_ERROR;
        host.free_recorded = true;
    }
    if (cudaEventRecord(fortai_cuda_event_handle(events->primary_done),
            context->bridge_stream[0]) != cudaSuccess ||
        cudaStreamWaitEvent(context->consumer_stream[0],
            fortai_cuda_event_handle(events->primary_done), 0) != cudaSuccess)
        return FORTAI_CUDA_RUNTIME_ERROR;
    return FORTAI_CUDA_OK;
}

static int fortai_cuda_q4_wait_output_buffer(fortai_cuda_q4_context *context,
    ggml_backend_t backend) {
    if (context == nullptr || backend == nullptr) return FORTAI_CUDA_INVALID;
    if (context->last_output_free != nullptr) {
        const int slot = fortai_cuda_q4_consumer_slot(context, context->device_ids[1]);
        if (fortai_cuda_q4_wait_scheduler_event(context, slot, context->last_output_free) != FORTAI_CUDA_OK)
            return FORTAI_CUDA_RUNTIME_ERROR;
    }
    return FORTAI_CUDA_OK;
}

static int fortai_cuda_q4_matvec_device_swiglu_impl(
    fortai_cuda_q4_context *context, const fortai_cuda_q4_weights *gate_weights,
    const fortai_cuda_q4_weights *up_weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements,
    bool remote_output) {
    if (context == nullptr || gate_weights == nullptr || up_weights == nullptr ||
        device_activation == nullptr || device_output == nullptr || activation_elements == 0)
        return FORTAI_CUDA_INVALID;
    fortai_cuda_q4_swiglu_plan *plan = fortai_cuda_q4_find_swiglu_plan(
        context, gate_weights, up_weights);
    if (plan == nullptr || plan->activation == nullptr || plan->output == nullptr) {
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    if (activation_elements != static_cast<size_t>(plan->activation->ne[0]) ||
        output_elements < static_cast<size_t>(plan->output->ne[0])) {
        return FORTAI_CUDA_INVALID;
    }

    const int primary_device = context->device_ids[0];
    const int weight_device = gate_weights->device;
    const bool remote = weight_device != primary_device;
    if (weight_device != primary_device && weight_device != context->device_ids[1])
        return FORTAI_CUDA_INVALID;
    if (remote_output != remote) {
        return FORTAI_CUDA_INVALID;
    }

    const size_t activation_bytes = activation_elements * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(plan->output->ne[0]) * sizeof(float);
    const bool async_bridge = remote && !context->sync_bridge_requested &&
        context->consumer_stream[0] != nullptr && context->consumer_stream[1] != nullptr;
    fortai_cuda_q4_bridge_events *bridge_events = nullptr;
    void *activation_ptr = const_cast<void *>(device_activation);
    void *output_ptr = device_output;
    if (remote) {
        const int code = fortai_cuda_q4_ensure_bridge(context, 1, activation_bytes,
            remote_output ? 0 : output_bytes);
        if (code != FORTAI_CUDA_OK) return code;
        if (async_bridge) {
            if (fortai_cuda_q4_queue_activation_remote(context, device_activation,
                    activation_bytes, &bridge_events) != FORTAI_CUDA_OK)
                return FORTAI_CUDA_RUNTIME_ERROR;
            if (!remote_output && fortai_cuda_q4_wait_output_buffer(context,
                    plan->backend) != FORTAI_CUDA_OK)
                return FORTAI_CUDA_RUNTIME_ERROR;
        } else {
            if (fortai_cuda_q4_prepare_input(context, primary_device) != FORTAI_CUDA_OK)
                return FORTAI_CUDA_RUNTIME_ERROR;
            if (fortai_cuda_q4_copy_between_devices(context, context->bridge_activation[1],
                    weight_device, device_activation, primary_device, activation_bytes) != FORTAI_CUDA_OK)
                return FORTAI_CUDA_RUNTIME_ERROR;
        }
        activation_ptr = context->bridge_activation[1];
        if (!remote_output) output_ptr = context->bridge_output[1];
    } else if (fortai_cuda_q4_prepare_input(context, primary_device) != FORTAI_CUDA_OK) {
        return FORTAI_CUDA_RUNTIME_ERROR;
    }

    void *saved_activation = plan->activation->data;
    void *saved_output = plan->output->data;
    plan->activation->data = activation_ptr;
    plan->output->data = output_ptr;
    enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context, plan->backend, plan->graph,
        &plan->cached_graph, &plan->cached_exec, &plan->graph_warmup, &plan->graph_disabled);
    plan->activation->data = saved_activation;
    plan->output->data = saved_output;
    if (status != GGML_STATUS_SUCCESS) {
        return FORTAI_CUDA_RUNTIME_ERROR;
    }

    if (remote) {
        if (async_bridge) {
            if (fortai_cuda_q4_record_remote_done(context, plan->backend, bridge_events) !=
                    FORTAI_CUDA_OK)
                return FORTAI_CUDA_RUNTIME_ERROR;
            if (remote_output) {
                if (cudaSetDevice(weight_device) != cudaSuccess ||
                    cudaStreamWaitEvent(context->consumer_stream[1],
                        fortai_cuda_event_handle(bridge_events->remote_done), 0) != cudaSuccess)
                    return FORTAI_CUDA_RUNTIME_ERROR;
            } else {
                const void *sources[1] = {output_ptr};
                void *destinations[1] = {device_output};
                const size_t bytes[1] = {output_bytes};
                if (fortai_cuda_q4_queue_outputs_primary(context, bridge_events, sources,
                        destinations, bytes, 1) != FORTAI_CUDA_OK)
                    return FORTAI_CUDA_RUNTIME_ERROR;
            }
        } else {
            ggml_backend_synchronize(plan->backend);
            if (!remote_output && fortai_cuda_q4_copy_between_devices(context, device_output,
                    primary_device, output_ptr, weight_device, output_bytes) != FORTAI_CUDA_OK)
                return FORTAI_CUDA_RUNTIME_ERROR;
        }
    } else if (fortai_cuda_q4_publish_output(context, plan->backend, primary_device) !=
            FORTAI_CUDA_OK) {
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_matvec_device_swiglu(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *gate_weights, const fortai_cuda_q4_weights *up_weights,
    const void *device_activation, size_t activation_elements, void *device_output,
    size_t output_elements) {
    fortai_cuda_q4_debug("enter swiglu");
    return fortai_cuda_q4_matvec_device_swiglu_impl(context, gate_weights, up_weights,
        device_activation, activation_elements, device_output, output_elements, false);
}

int fortai_cuda_q4_matvec_device_swiglu_remote_output(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *gate_weights, const fortai_cuda_q4_weights *up_weights,
    const void *device_activation, size_t activation_elements, void *device_output,
    size_t output_elements) {
    fortai_cuda_q4_debug("enter swiglu remote output");
    return fortai_cuda_q4_matvec_device_swiglu_impl(context, gate_weights, up_weights,
        device_activation, activation_elements, device_output, output_elements, true);
}

/* Run a complete Q4 FFN graph without handing its intermediate SwiGLU vector
 * back to the native stream.  GGML-CUDA still selects the format-specific
 * fused matmul kernels, while one graph launch covers gate/up, SwiGLU, and
 * down.  The stream bridge is crossed only once for the complete FFN. */
int fortai_cuda_q4_matvec_device_swiglu_down(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *gate_weights, const fortai_cuda_q4_weights *up_weights,
    const fortai_cuda_q4_weights *down_weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    fortai_cuda_q4_debug("enter swiglu down");
    if (context == nullptr || gate_weights == nullptr || up_weights == nullptr ||
        down_weights == nullptr || device_activation == nullptr || device_output == nullptr ||
        activation_elements == 0 || gate_weights->owner != context || up_weights->owner != context ||
        down_weights->owner != context || gate_weights->device != up_weights->device ||
        gate_weights->device != down_weights->device)
        return FORTAI_CUDA_INVALID;
    fortai_cuda_q4_ffn_plan *plan = fortai_cuda_q4_find_ffn_plan(
        context, gate_weights, up_weights, down_weights);
    if (plan == nullptr || plan->activation == nullptr || plan->swiglu_output == nullptr ||
        plan->output == nullptr)
        return FORTAI_CUDA_RUNTIME_ERROR;
    if (activation_elements != static_cast<size_t>(plan->activation->ne[0]) ||
        output_elements < static_cast<size_t>(plan->output->ne[0]))
        return FORTAI_CUDA_INVALID;

    const int primary_device = context->device_ids[0];
    const int weight_device = gate_weights->device;
    const bool remote = weight_device != primary_device;
    if (weight_device != primary_device && weight_device != context->device_ids[1])
        return FORTAI_CUDA_INVALID;
    const size_t activation_bytes = activation_elements * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(plan->output->ne[0]) * sizeof(float);
    const bool async_bridge = remote && !context->sync_bridge_requested &&
        context->consumer_stream[0] != nullptr && context->consumer_stream[1] != nullptr;
    fortai_cuda_q4_bridge_events *bridge_events = nullptr;
    void *activation_ptr = const_cast<void *>(device_activation);
    void *output_ptr = device_output;
    if (remote) {
        const int code = fortai_cuda_q4_ensure_bridge(context, 1, activation_bytes, output_bytes);
        if (code != FORTAI_CUDA_OK) {
            fortai_cuda_q4_debug_code("swiglu-down ensure bridge", code);
            return code;
        }
        if (async_bridge) {
            if (fortai_cuda_q4_queue_activation_remote(context, device_activation,
                    activation_bytes, &bridge_events) != FORTAI_CUDA_OK ||
                fortai_cuda_q4_wait_output_buffer(context, plan->backend) != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("swiglu-down queue activation/output");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        } else {
            if (fortai_cuda_q4_prepare_input(context, primary_device) != FORTAI_CUDA_OK ||
                fortai_cuda_q4_copy_between_devices(context, context->bridge_activation[1],
                    weight_device, device_activation, primary_device, activation_bytes) !=
                    FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("swiglu-down sync activation");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        }
        activation_ptr = context->bridge_activation[1];
        if (async_bridge) output_ptr = context->bridge_output[1];
        else output_ptr = context->bridge_output[1];
    } else if (fortai_cuda_q4_prepare_input(context, primary_device) != FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug("swiglu-down local prepare");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }

    void *saved_activation = plan->activation->data;
    void *saved_output = plan->output->data;
    plan->activation->data = activation_ptr;
    plan->output->data = output_ptr;
    enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context, plan->backend, plan->graph,
        &plan->cached_graph, &plan->cached_exec, &plan->graph_warmup, &plan->graph_disabled);
    plan->activation->data = saved_activation;
    plan->output->data = saved_output;
    if (status != GGML_STATUS_SUCCESS) {
        fortai_cuda_q4_debug("GGML Q4 matvec graph");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }

    if (remote) {
        if (async_bridge) {
            if (fortai_cuda_q4_record_remote_done(context, plan->backend, bridge_events) !=
                    FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("swiglu-down remote done");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
            const void *sources[1] = {output_ptr};
            void *destinations[1] = {device_output};
            const size_t bytes[1] = {output_bytes};
            if (fortai_cuda_q4_queue_outputs_primary(context, bridge_events, sources,
                    destinations, bytes, 1) != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("swiglu-down queue output");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        } else {
            ggml_backend_synchronize(down_weights->backend);
            if (fortai_cuda_q4_copy_between_devices(context, device_output, primary_device,
                    output_ptr, weight_device, output_bytes) != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("swiglu-down sync output");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        }
    } else if (fortai_cuda_q4_publish_output(context, plan->backend, primary_device) !=
            FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug("swiglu-down local publish");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    return FORTAI_CUDA_OK;
}

int fortai_cuda_q4_matvec_device(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    fortai_cuda_q4_debug("enter matvec");
    if (context == nullptr || weights == nullptr || weights->owner != context || weights->weight == nullptr ||
        device_activation == nullptr || device_output == nullptr || activation_elements == 0 ||
        output_elements < static_cast<size_t>(weights->output->ne[0]))
        return FORTAI_CUDA_INVALID;

    /* The cached GGML graph owns the tensor descriptors, but its F32 edge can
     * be temporarily rebound to FortAI's resident activation/output.  The
     * descriptors are restored on every exit path before the next scheduler
     * invocation; no allocator-owned edge copy is needed on this path. */
    if (activation_elements != static_cast<size_t>(weights->activation->ne[0]))
        return FORTAI_CUDA_INVALID;
    const size_t activation_bytes = activation_elements * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(weights->output->ne[0]) * sizeof(float);
    const int primary_device = context->device_ids[0];
    const bool remote = weights->device != primary_device;

    int bridge_slot = -1;
    void *activation_ptr = const_cast<void *>(device_activation);
    void *output_ptr = device_output;
    fortai_cuda_q4_bridge_events *bridge_events = nullptr;
    if (remote) {
        if (weights->device != context->device_ids[1]) return FORTAI_CUDA_INVALID;
        bridge_slot = 1;
        int code = fortai_cuda_q4_ensure_bridge(context, bridge_slot, activation_bytes, output_bytes);
        if (code != FORTAI_CUDA_OK) {
            fortai_cuda_q4_debug_code("matvec ensure bridge", code);
            return code;
        }
        if (context->consumer_stream[0] != nullptr && context->consumer_stream[1] != nullptr) {
            if (fortai_cuda_q4_queue_activation_remote(context, device_activation,
                    activation_bytes, &bridge_events) != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("matvec queue input");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
            if (fortai_cuda_q4_wait_output_buffer(context, weights->backend) != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("matvec wait output");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
            activation_ptr = context->bridge_activation[bridge_slot];
            output_ptr = context->bridge_output[bridge_slot];
        } else {
            /* Standalone callers have no producer stream to attach to.  Keep
             * the conservative synchronous path for that ABI. */
            if (fortai_cuda_q4_copy_between_devices(context, context->bridge_activation[bridge_slot], weights->device,
                    device_activation, primary_device, activation_bytes) != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("matvec sync input");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
            activation_ptr = context->bridge_activation[bridge_slot];
            output_ptr = context->bridge_output[bridge_slot];
        }
    } else if (fortai_cuda_q4_prepare_input(context, primary_device) != FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug("matvec local prepare");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    void *saved_activation = weights->activation->data;
    void *saved_output = weights->output->data;
    auto restore_edges = [&]() {
        weights->activation->data = saved_activation;
        weights->output->data = saved_output;
    };
    weights->activation->data = activation_ptr;
    weights->output->data = output_ptr;
    enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context, weights->backend, weights->graph,
        &const_cast<fortai_cuda_q4_weights *>(weights)->cached_graph,
        &const_cast<fortai_cuda_q4_weights *>(weights)->cached_exec,
        &const_cast<fortai_cuda_q4_weights *>(weights)->graph_warmup,
        &const_cast<fortai_cuda_q4_weights *>(weights)->graph_disabled);
    if (status == GGML_STATUS_SUCCESS && !remote &&
        fortai_cuda_q4_publish_output(context, weights->backend, weights->device) != FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug("matvec local publish");
        status = GGML_STATUS_FAILED;
    }
    if (status != GGML_STATUS_SUCCESS)
        fortai_cuda_q4_debug("matvec graph");
    restore_edges();
    if (status == GGML_STATUS_SUCCESS && remote) {
        if (bridge_events != nullptr) {
            if (fortai_cuda_q4_record_remote_done(context, weights->backend, bridge_events) != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("matvec remote done");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
            const void *sources[1] = {output_ptr};
            void *destinations[1] = {device_output};
            const size_t bytes[1] = {output_bytes};
            if (fortai_cuda_q4_queue_outputs_primary(context, bridge_events, sources, destinations,
                    bytes, 1) != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("matvec queue output");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        } else {
            ggml_backend_synchronize(weights->backend);
            if (fortai_cuda_q4_copy_between_devices(context, device_output, primary_device, output_ptr,
                    weights->device, output_bytes) != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("matvec sync output");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        }
    }
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
    fortai_cuda_q4_debug("enter group");
    if (context == nullptr || weights == nullptr || device_activation == nullptr ||
        device_outputs == nullptr || output_elements == nullptr || count < 1 || count > 8 ||
        activation_elements == 0) return FORTAI_CUDA_INVALID;

    const size_t activation_bytes = activation_elements * sizeof(float);
    const int primary_device = context->device_ids[0];
    for (int i = 0; i < count; ++i) {
        if (weights[i] == nullptr || weights[i]->owner != context || device_outputs[i] == nullptr ||
            output_elements[i] < static_cast<size_t>(weights[i]->output->ne[0]) ||
            activation_elements != static_cast<size_t>(weights[i]->activation->ne[0]))
            return FORTAI_CUDA_INVALID;
    }

    bool has_primary = false;
    for (int i = 0; i < count; ++i)
        if (weights[i]->device == primary_device) has_primary = true;
    if (has_primary && fortai_cuda_q4_prepare_input(context, primary_device) != FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug("group primary prepare");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }

    bool done[8] = {};
    for (int first_index = 0; first_index < count; ++first_index) {
        if (done[first_index]) continue;
        const int device = weights[first_index]->device;
        if (device != primary_device && device != context->device_ids[1])
            return FORTAI_CUDA_INVALID;

        int members[8] = {};
        int member_count = 0;
        for (int i = first_index; i < count; ++i) {
            if (!done[i] && weights[i]->device == device)
                members[member_count++] = i;
        }
        if (member_count == 0) continue;
        for (int i = 0; i < member_count; ++i) done[members[i]] = true;

        const bool remote = device != primary_device;
        void * activation_ptr = const_cast<void *>(device_activation);
        fortai_cuda_q4_bridge_events *bridge_events = nullptr;
        const bool async_bridge = remote && !context->sync_bridge_requested &&
            context->consumer_stream[0] != nullptr && context->consumer_stream[1] != nullptr;
        if (remote) {
            int code = fortai_cuda_q4_ensure_bridge(context, 1, activation_bytes, 0);
            if (code != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug_code("group ensure input bridge", code);
                return code;
            }
            if (async_bridge) {
                if (fortai_cuda_q4_queue_activation_remote(context, device_activation,
                        activation_bytes, &bridge_events) != FORTAI_CUDA_OK) {
                    fortai_cuda_q4_debug("group queue input");
                    return FORTAI_CUDA_RUNTIME_ERROR;
                }
            } else if (fortai_cuda_q4_copy_between_devices(context, context->bridge_activation[1], device,
                    device_activation, primary_device, activation_bytes) != FORTAI_CUDA_OK) {
                    fortai_cuda_q4_debug("group sync input");
                    return FORTAI_CUDA_RUNTIME_ERROR;
            }
            activation_ptr = context->bridge_activation[1];
        }

        fortai_cuda_q4_group_plan *plan =
            fortai_cuda_q4_find_device_group_plan(context, weights, members, member_count);
        if (plan == nullptr) {
            fortai_cuda_q4_debug("group plan");
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        size_t bridge_output_bytes = 0;
        size_t output_offsets[8] = {};
        if (remote) {
            for (int i = 0; i < member_count; ++i) {
                output_offsets[i] = bridge_output_bytes;
                bridge_output_bytes += static_cast<size_t>(weights[members[i]]->output->ne[0]) * sizeof(float);
            }
            int code = fortai_cuda_q4_ensure_bridge(context, 1, activation_bytes, bridge_output_bytes);
            if (code != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug_code("group ensure output bridge", code);
                return code;
            }
            if (async_bridge && fortai_cuda_q4_wait_output_buffer(context, plan->backend) != FORTAI_CUDA_OK) {
                fortai_cuda_q4_debug("group wait output");
                return FORTAI_CUDA_RUNTIME_ERROR;
            }
        }
        void *saved_activation[8] = {};
        void *saved_output[8] = {};
        for (int i = 0; i < member_count; ++i) {
            const int index = members[i];
            saved_activation[i] = weights[index]->activation->data;
            saved_output[i] = weights[index]->output->data;
            weights[index]->activation->data = activation_ptr;
            weights[index]->output->data = remote ?
                static_cast<void *>(static_cast<char *>(context->bridge_output[1]) + output_offsets[i]) :
                device_outputs[index];
        }
        auto restore_edges = [&]() {
            for (int i = 0; i < member_count; ++i) {
                const int index = members[i];
                weights[index]->activation->data = saved_activation[i];
                weights[index]->output->data = saved_output[i];
            }
        };
        enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context, plan->backend, plan->graph,
            &plan->cached_graph, &plan->cached_exec, &plan->graph_warmup, &plan->graph_disabled);
        if (status == GGML_STATUS_SUCCESS && !remote &&
            fortai_cuda_q4_publish_output(context, plan->backend, device) != FORTAI_CUDA_OK)
            status = GGML_STATUS_FAILED;
        if (status != GGML_STATUS_SUCCESS) {
            fortai_cuda_q4_debug("group graph");
            restore_edges();
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
        restore_edges();
        if (remote) {
            if (async_bridge) {
                if (fortai_cuda_q4_record_remote_done(context, plan->backend, bridge_events) != FORTAI_CUDA_OK) {
                    fortai_cuda_q4_debug("group remote done");
                    return FORTAI_CUDA_RUNTIME_ERROR;
                }
                const void *sources[8] = {};
                void *destinations[8] = {};
                size_t bytes[8] = {};
                for (int i = 0; i < member_count; ++i) {
                    const int index = members[i];
                    sources[i] = static_cast<char *>(context->bridge_output[1]) + output_offsets[i];
                    destinations[i] = device_outputs[index];
                    bytes[i] = static_cast<size_t>(weights[index]->output->ne[0]) * sizeof(float);
                }
                if (fortai_cuda_q4_queue_outputs_primary(context, bridge_events, sources,
                        destinations, bytes, member_count) != FORTAI_CUDA_OK) {
                    fortai_cuda_q4_debug("group queue output");
                    return FORTAI_CUDA_RUNTIME_ERROR;
                }
            } else {
                ggml_backend_synchronize(plan->backend);
                for (int i = 0; i < member_count; ++i) {
                    const int index = members[i];
                    const size_t bytes = static_cast<size_t>(weights[index]->output->ne[0]) * sizeof(float);
                    if (fortai_cuda_q4_copy_between_devices(context, device_outputs[index], primary_device,
                            static_cast<char *>(context->bridge_output[1]) + output_offsets[i], device, bytes) !=
                            FORTAI_CUDA_OK)
                        return FORTAI_CUDA_RUNTIME_ERROR;
                }
            }
        }
    }
    return FORTAI_CUDA_OK;
}

/* Four recurrent projections (QKV/conv, gate, alpha, beta) consume the same
 * normalized activation.  Keeping them in one graph avoids an otherwise
 * unavoidable scheduler launch and activation quantization boundary between
 * the two legacy pair calls. */
int fortai_cuda_q4_matvec_device_quad(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *first_weights,
    const fortai_cuda_q4_weights *second_weights,
    const fortai_cuda_q4_weights *third_weights,
    const fortai_cuda_q4_weights *fourth_weights, const void *device_activation,
    size_t activation_elements, void *first_output, size_t first_output_elements,
    void *second_output, size_t second_output_elements, void *third_output,
    size_t third_output_elements, void *fourth_output, size_t fourth_output_elements) {
    const fortai_cuda_q4_weights *weights[4] = {
        first_weights, second_weights, third_weights, fourth_weights};
    void *outputs[4] = {first_output, second_output, third_output, fourth_output};
    const size_t output_elements[4] = {first_output_elements, second_output_elements,
        third_output_elements, fourth_output_elements};
    return fortai_cuda_q4_matvec_device_group(context, weights, device_activation,
        activation_elements, outputs, output_elements, 4);
}

static fortai_cuda_q4_batch_plan *fortai_cuda_q4_find_batch_plan(
    fortai_cuda_q4_context *context, const fortai_cuda_q4_weights * const *weights,
    int count, int batch) {
    if (context == nullptr || weights == nullptr || count < 1 || count > 3 || batch <= 0)
        return nullptr;
    for (auto &candidate : context->batch_plans) {
        if (candidate.count != count || candidate.batch != batch ||
            candidate.backend != weights[0]->backend) continue;
        bool match = true;
        for (int i = 0; i < count; ++i)
            if (candidate.weights[i] != weights[i]) match = false;
        if (match) return &candidate;
    }

    fortai_cuda_q4_batch_plan candidate;
    candidate.count = count;
    candidate.batch = batch;
    candidate.backend = weights[0]->backend;
    ggml_init_params params{};
    params.mem_size = 4 << 20;
    params.no_alloc = true;
    candidate.context = ggml_init(params);
    if (candidate.context == nullptr) return nullptr;
    /* The immutable quantized allocation belongs to the normal per-tensor
     * plan.  The batch context owns only descriptors and its activation/
     * output compute buffer, so model VRAM is not duplicated. */
    for (int i = 0; i < count; ++i) {
        if (weights[i] == nullptr || weights[i]->weight == nullptr ||
            weights[i]->backend != candidate.backend ||
            weights[i]->device != weights[0]->device ||
            weights[i]->weight->ne[0] != weights[0]->weight->ne[0]) {
            ggml_free(candidate.context);
            return nullptr;
        }
        candidate.weights[i] = weights[i];
        candidate.weight[i] = ggml_new_tensor_2d(candidate.context, weights[i]->weight->type,
            weights[i]->weight->ne[0], weights[i]->weight->ne[1]);
        if (candidate.weight[i] == nullptr) {
            ggml_free(candidate.context);
            return nullptr;
        }
        candidate.weight[i]->data = weights[i]->weight->data;
        candidate.weight[i]->buffer = weights[i]->weight->buffer;
    }
    candidate.activation[0] = ggml_new_tensor_2d(candidate.context, GGML_TYPE_F32,
        weights[0]->weight->ne[0], batch);
    if (candidate.activation[0] == nullptr) {
        ggml_free(candidate.context);
        return nullptr;
    }
    for (int i = 1; i < count; ++i) candidate.activation[i] = candidate.activation[0];
    for (int i = 0; i < count; ++i) {
        candidate.output[i] = ggml_mul_mat(candidate.context, candidate.weight[i], candidate.activation[0]);
        if (candidate.output[i] == nullptr) {
            ggml_free(candidate.context);
            return nullptr;
        }
    }
    candidate.graph = ggml_new_graph_custom(candidate.context, GGML_DEFAULT_GRAPH_SIZE, false);
    if (candidate.graph == nullptr) {
        ggml_free(candidate.context);
        return nullptr;
    }
    for (int i = 0; i < count; ++i) ggml_build_forward_expand(candidate.graph, candidate.output[i]);
    candidate.buffer = ggml_backend_alloc_ctx_tensors(candidate.context, candidate.backend);
    if (candidate.buffer == nullptr) {
        ggml_free(candidate.context);
        return nullptr;
    }
    ggml_backend_buffer_set_usage(candidate.buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    try {
        context->batch_plans.push_back(candidate);
    } catch (...) {
        ggml_backend_buffer_free(candidate.buffer);
        ggml_free(candidate.context);
        return nullptr;
    }
    return &context->batch_plans.back();
}

static fortai_cuda_q4_embedding_batch_plan *fortai_cuda_q4_find_embedding_batch_plan(
    fortai_cuda_q4_context *context, const fortai_cuda_q4_weights *weights, int batch) {
    if (context == nullptr || weights == nullptr || weights->owner != context ||
        weights->weight == nullptr || weights->backend == nullptr || batch <= 0)
        return nullptr;
    for (auto &candidate : context->embedding_batch_plans) {
        if (candidate.weights == weights && candidate.batch == batch)
            return &candidate;
    }

    fortai_cuda_q4_embedding_batch_plan candidate;
    candidate.weights = weights;
    candidate.batch = batch;
    candidate.backend = weights->backend;
    ggml_init_params params{};
    params.mem_size = std::max<size_t>(2u << 20, static_cast<size_t>(batch) * sizeof(int32_t) * 8);
    params.no_alloc = true;
    candidate.context = ggml_init(params);
    if (candidate.context == nullptr) return nullptr;
    candidate.weight = ggml_new_tensor_2d(candidate.context, weights->weight->type,
        weights->weight->ne[0], weights->weight->ne[1]);
    candidate.ids = ggml_new_tensor_1d(candidate.context, GGML_TYPE_I32, batch);
    if (candidate.weight == nullptr || candidate.ids == nullptr) {
        ggml_free(candidate.context);
        return nullptr;
    }
    candidate.weight->data = weights->weight->data;
    candidate.weight->buffer = weights->weight->buffer;
    candidate.output = ggml_get_rows(candidate.context, candidate.weight, candidate.ids);
    candidate.graph = ggml_new_graph_custom(candidate.context, GGML_DEFAULT_GRAPH_SIZE, false);
    if (candidate.output == nullptr || candidate.graph == nullptr) {
        ggml_free(candidate.context);
        return nullptr;
    }
    ggml_build_forward_expand(candidate.graph, candidate.output);
    candidate.buffer = ggml_backend_alloc_ctx_tensors(candidate.context, candidate.backend);
    if (candidate.buffer == nullptr || candidate.ids->data == nullptr || candidate.output->data == nullptr) {
        if (candidate.buffer != nullptr) ggml_backend_buffer_free(candidate.buffer);
        ggml_free(candidate.context);
        return nullptr;
    }
    ggml_backend_buffer_set_usage(candidate.buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    try {
        context->embedding_batch_plans.push_back(candidate);
    } catch (...) {
        ggml_backend_buffer_free(candidate.buffer);
        ggml_free(candidate.context);
        return nullptr;
    }
    return &context->embedding_batch_plans.back();
}

int fortai_cuda_q4_matmul_device_group(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights * const *weights, const void *device_activation,
    size_t activation_elements, int batch, void * const *device_outputs,
    const size_t *output_elements, int count) {
    if (context == nullptr || weights == nullptr || device_activation == nullptr ||
        device_outputs == nullptr || output_elements == nullptr || count < 1 || count > 3 ||
        batch <= 0 || activation_elements == 0) return FORTAI_CUDA_INVALID;
    const int device = weights[0] != nullptr ? weights[0]->device : -1;
    const int primary = context->device_ids[0];
    if (device != primary) return FORTAI_CUDA_INVALID;
    for (int i = 0; i < count; ++i) {
        if (weights[i] == nullptr || weights[i]->owner != context ||
            weights[i]->device != device || device_outputs[i] == nullptr ||
            output_elements[i] < static_cast<size_t>(weights[i]->output->ne[0]) * batch ||
            activation_elements != static_cast<size_t>(weights[i]->weight->ne[0]) * batch)
            return FORTAI_CUDA_INVALID;
    }
    if (fortai_cuda_q4_prepare_input(context, primary) != FORTAI_CUDA_OK)
        return FORTAI_CUDA_RUNTIME_ERROR;
    fortai_cuda_q4_batch_plan *plan = fortai_cuda_q4_find_batch_plan(context, weights, count, batch);
    if (plan == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
    void *saved_activation = plan->activation[0]->data;
    void *saved_output[3] = {nullptr, nullptr, nullptr};
    plan->activation[0]->data = const_cast<void *>(device_activation);
    for (int i = 0; i < count; ++i) {
        saved_output[i] = plan->output[i]->data;
        plan->output[i]->data = device_outputs[i];
    }
    const enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context, plan->backend, plan->graph,
        &plan->cached_graph, &plan->cached_exec, &plan->graph_warmup, &plan->graph_disabled);
    if (status == GGML_STATUS_SUCCESS &&
        fortai_cuda_q4_publish_output(context, plan->backend, device) != FORTAI_CUDA_OK) {
        plan->activation[0]->data = saved_activation;
        for (int i = 0; i < count; ++i) plan->output[i]->data = saved_output[i];
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    plan->activation[0]->data = saved_activation;
    for (int i = 0; i < count; ++i) plan->output[i]->data = saved_output[i];
    return status == GGML_STATUS_SUCCESS ? FORTAI_CUDA_OK : FORTAI_CUDA_RUNTIME_ERROR;
}

int fortai_cuda_q4_matmul_device(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *device_activation,
    size_t activation_elements, int batch, void *device_output,
    size_t output_elements) {
    if (context == nullptr || weights == nullptr || device_output == nullptr || batch <= 0)
        return FORTAI_CUDA_INVALID;
    const fortai_cuda_q4_weights *group[1] = {weights};
    void *outputs[1] = {device_output};
    const size_t sizes[1] = {output_elements};
    return fortai_cuda_q4_matmul_device_group(context, group, device_activation,
        activation_elements, batch, outputs, sizes, 1);
}

int fortai_cuda_q4_matvec_device_group_remote_output(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights * const *weights, const void *device_activation,
    size_t activation_elements, void * const *device_outputs,
    const size_t *output_elements, int count) {
    fortai_cuda_q4_debug("enter group remote output");
    if (context == nullptr || weights == nullptr || device_activation == nullptr ||
        device_outputs == nullptr || output_elements == nullptr || count < 1 || count > 3 ||
        activation_elements == 0 || context->device_ids[0] == context->device_ids[1])
        return FORTAI_CUDA_INVALID;

    const int primary_device = context->device_ids[0];
    const int remote_device = context->device_ids[1];
    const size_t activation_bytes = activation_elements * sizeof(float);
    for (int i = 0; i < count; ++i) {
        if (weights[i] == nullptr || weights[i]->owner != context ||
            weights[i]->device != remote_device || device_outputs[i] == nullptr ||
            output_elements[i] < static_cast<size_t>(weights[i]->output->ne[0]) ||
            activation_elements != static_cast<size_t>(weights[i]->activation->ne[0]))
            return FORTAI_CUDA_INVALID;
    }

    /* The input is produced by FortAI's primary stream.  Queue the PCIe
     * staging edge and leave the projection output on the remote device for
     * the attention core. */
    const bool async_bridge = !context->sync_bridge_requested &&
        context->consumer_stream[0] != nullptr && context->consumer_stream[1] != nullptr;
    fortai_cuda_q4_bridge_events *bridge_events = nullptr;
    if (async_bridge) {
        if (fortai_cuda_q4_queue_activation_remote(context, device_activation,
                activation_bytes, &bridge_events) != FORTAI_CUDA_OK) {
            fortai_cuda_q4_debug("remote-output queue input");
            return FORTAI_CUDA_RUNTIME_ERROR;
        }
    } else if (fortai_cuda_q4_prepare_input(context, primary_device) != FORTAI_CUDA_OK) {
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    int code = fortai_cuda_q4_ensure_bridge(context, 1, activation_bytes, 0);
    if (code != FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug_code("remote-output ensure bridge", code);
        return code;
    }
    if (!async_bridge && fortai_cuda_q4_copy_between_devices(context, context->bridge_activation[1], remote_device,
            device_activation, primary_device, activation_bytes) != FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug("remote-output sync input");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }

    int members[3] = {0, 1, 2};
    fortai_cuda_q4_group_plan *plan =
        fortai_cuda_q4_find_device_group_plan(context, weights, members, count);
    if (plan == nullptr) {
        fortai_cuda_q4_debug("remote-output group plan");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    if (async_bridge && fortai_cuda_q4_wait_output_buffer(context, plan->backend) != FORTAI_CUDA_OK) {
        fortai_cuda_q4_debug("remote-output wait output");
        return FORTAI_CUDA_RUNTIME_ERROR;
    }
    void *saved_activation[3] = {nullptr, nullptr, nullptr};
    void *saved_output[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < count; ++i) {
        saved_activation[i] = weights[i]->activation->data;
        saved_output[i] = weights[i]->output->data;
        weights[i]->activation->data = context->bridge_activation[1];
        weights[i]->output->data = device_outputs[i];
    }
    enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context, plan->backend, plan->graph,
        &plan->cached_graph, &plan->cached_exec, &plan->graph_warmup, &plan->graph_disabled);
    if (status == GGML_STATUS_SUCCESS && async_bridge) {
        if (fortai_cuda_q4_record_remote_done(context, plan->backend, bridge_events) != FORTAI_CUDA_OK) {
            fortai_cuda_q4_debug("remote-output remote done");
            status = GGML_STATUS_FAILED;
        }
        else if (cudaSetDevice(remote_device) != cudaSuccess ||
            cudaStreamWaitEvent(context->consumer_stream[1],
                fortai_cuda_event_handle(bridge_events->remote_done), 0) != cudaSuccess) {
            fortai_cuda_q4_debug("remote-output consumer wait");
            status = GGML_STATUS_FAILED;
        }
    } else if (status == GGML_STATUS_SUCCESS) {
        /* Standalone callers retain the conservative host-visible barrier. */
        if (fortai_cuda_q4_publish_output(context, plan->backend, remote_device) != FORTAI_CUDA_OK)
            status = GGML_STATUS_FAILED;
    }
    for (int i = 0; i < count; ++i) {
        weights[i]->activation->data = saved_activation[i];
        weights[i]->output->data = saved_output[i];
    }
    if (status != GGML_STATUS_SUCCESS) fortai_cuda_q4_debug("remote-output graph");
    return status == GGML_STATUS_SUCCESS ? FORTAI_CUDA_OK : FORTAI_CUDA_RUNTIME_ERROR;
}

int fortai_cuda_q4_matvec_device_remote_input(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const void *device_activation,
    size_t activation_elements, void *device_output, size_t output_elements) {
    fortai_cuda_q4_debug("enter remote input");
    if (context == nullptr || weights == nullptr || weights->owner != context ||
        device_activation == nullptr || device_output == nullptr || activation_elements == 0 ||
        context->device_ids[0] == context->device_ids[1] ||
        weights->device != context->device_ids[1] ||
        output_elements < static_cast<size_t>(weights->output->ne[0]) ||
        activation_elements != static_cast<size_t>(weights->activation->ne[0]))
        return FORTAI_CUDA_INVALID;

    const int primary_device = context->device_ids[0];
    const int remote_device = context->device_ids[1];
    const size_t activation_bytes = activation_elements * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(weights->output->ne[0]) * sizeof(float);

    /* The attention core and this Q4 graph use different stream owners.  The
     * attached second-device consumer stream turns the dependency into a
     * device-side event wait; no host download of the attention work is
     * needed before launching the projection. */
    if (fortai_cuda_q4_prepare_input(context, remote_device) != FORTAI_CUDA_OK)
        return FORTAI_CUDA_RUNTIME_ERROR;
    int code = fortai_cuda_q4_ensure_bridge(context, 1, activation_bytes, output_bytes);
    if (code != FORTAI_CUDA_OK) return code;
    const bool async_bridge = !context->sync_bridge_requested &&
        context->consumer_stream[0] != nullptr && context->consumer_stream[1] != nullptr;
    fortai_cuda_q4_bridge_events *bridge_events = nullptr;
    if (async_bridge) {
        if (fortai_cuda_q4_next_bridge_events(context, &bridge_events) != FORTAI_CUDA_OK)
            return FORTAI_CUDA_RUNTIME_ERROR;
        if (fortai_cuda_q4_wait_output_buffer(context, weights->backend) != FORTAI_CUDA_OK)
            return FORTAI_CUDA_RUNTIME_ERROR;
    }
    void *saved_activation = weights->activation->data;
    void *saved_output = weights->output->data;
    weights->activation->data = const_cast<void *>(device_activation);
    weights->output->data = context->bridge_output[1];
    enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context, weights->backend, weights->graph,
        &const_cast<fortai_cuda_q4_weights *>(weights)->cached_graph,
        &const_cast<fortai_cuda_q4_weights *>(weights)->cached_exec,
        &const_cast<fortai_cuda_q4_weights *>(weights)->graph_warmup,
        &const_cast<fortai_cuda_q4_weights *>(weights)->graph_disabled);
    weights->activation->data = saved_activation;
    weights->output->data = saved_output;
    if (status != GGML_STATUS_SUCCESS) return FORTAI_CUDA_RUNTIME_ERROR;
    if (async_bridge) {
        if (fortai_cuda_q4_record_remote_done(context, weights->backend, bridge_events) != FORTAI_CUDA_OK)
            return FORTAI_CUDA_RUNTIME_ERROR;
        const void *sources[1] = {context->bridge_output[1]};
        void *destinations[1] = {device_output};
        const size_t bytes[1] = {output_bytes};
        if (fortai_cuda_q4_queue_outputs_primary(context, bridge_events, sources, destinations,
                bytes, 1) != FORTAI_CUDA_OK)
            return FORTAI_CUDA_RUNTIME_ERROR;
    } else {
        ggml_backend_synchronize(weights->backend);
        if (fortai_cuda_q4_copy_between_devices(context, device_output, primary_device,
                context->bridge_output[1], remote_device, output_bytes) != FORTAI_CUDA_OK)
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
    context->embedding_id = static_cast<int32_t>(token_id);
    ggml_backend_tensor_set_async(weights->backend, weights->embedding_ids,
        &context->embedding_id, 0, sizeof(context->embedding_id));
    const int primary_device = context->device_ids[0];
    const bool remote = weights->device != primary_device;
    /* Token lookup writes directly into the native Q8 stream's activation
     * buffer.  When the preceding token deliberately skipped its output
     * projection (prompt fast path), make the Q4 scheduler wait for the
     * last Q8 producer before overwriting that buffer. */
    if (fortai_cuda_q4_prepare_input(context, weights->device) != FORTAI_CUDA_OK)
        return FORTAI_CUDA_RUNTIME_ERROR;
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
    enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context, weights->backend, weights->embedding_graph,
        &const_cast<fortai_cuda_q4_weights *>(weights)->embedding_cached_graph,
        &const_cast<fortai_cuda_q4_weights *>(weights)->embedding_cached_exec,
        &const_cast<fortai_cuda_q4_weights *>(weights)->embedding_graph_warmup,
        &const_cast<fortai_cuda_q4_weights *>(weights)->embedding_graph_disabled);
    weights->embedding_output->data = saved_output;
    if (status == GGML_STATUS_SUCCESS && !remote) {
        if (fortai_cuda_q4_publish_output(context, weights->backend, primary_device) != FORTAI_CUDA_OK)
            status = GGML_STATUS_FAILED;
    } else if (status == GGML_STATUS_SUCCESS && remote) {
        /* Remote token embeddings are supported for standalone callers.  The
         * native Qwen pipeline pins this tensor to the primary board, so this
         * conservative path is outside the decode hot loop. */
        ggml_backend_synchronize(weights->backend);
        if (fortai_cuda_q4_copy_between_devices(context, device_output, primary_device, output_ptr, weights->device,
                static_cast<size_t>(weights->weight->ne[0]) * sizeof(float)) != FORTAI_CUDA_OK)
            status = GGML_STATUS_FAILED;
    }
    return status == GGML_STATUS_SUCCESS ? FORTAI_CUDA_OK : FORTAI_CUDA_RUNTIME_ERROR;
}

int fortai_cuda_q4_embedding_device_batch(fortai_cuda_q4_context *context,
    const fortai_cuda_q4_weights *weights, const int32_t *host_tokens, int batch,
    void *device_output, size_t output_elements) {
    if (context == nullptr || weights == nullptr || weights->owner != context || weights->weight == nullptr ||
        host_tokens == nullptr || batch <= 0 || device_output == nullptr ||
        weights->device != context->device_ids[0] ||
        output_elements < static_cast<size_t>(weights->weight->ne[0]) * batch)
        return FORTAI_CUDA_INVALID;
    const int code = fortai_cuda_q4_prepare_input(context, weights->device);
    if (code != FORTAI_CUDA_OK) return code;
    auto *plan = fortai_cuda_q4_find_embedding_batch_plan(context, weights, batch);
    if (plan == nullptr) return FORTAI_CUDA_RUNTIME_ERROR;
    /* A synchronous set keeps the caller's short-lived token array from
     * being reused while GGML's CUDA stream is still reading it.  The copy is
     * only batch*sizeof(int32_t), and the expensive row lookup remains fully
     * asynchronous on the resident stream. */
    ggml_backend_tensor_set(plan->ids, host_tokens, 0, static_cast<size_t>(batch) * sizeof(int32_t));
    void *saved_output = plan->output->data;
    plan->output->data = device_output;
    const enum ggml_status status = fortai_cuda_q4_graph_compute_cached(context, plan->backend, plan->graph,
        &plan->cached_graph, &plan->cached_exec, &plan->graph_warmup, &plan->graph_disabled);
    plan->output->data = saved_output;
    if (status != GGML_STATUS_SUCCESS) return FORTAI_CUDA_RUNTIME_ERROR;
    return fortai_cuda_q4_publish_output(context, plan->backend, weights->device);
}
