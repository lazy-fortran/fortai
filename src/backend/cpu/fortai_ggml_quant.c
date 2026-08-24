/*
 * Small ABI adapter for the GGML reference dequantizers.  GGUF stores the
 * quantized blocks verbatim; this adapter resolves the format-specific
 * dequantizers at runtime so the Fortran runtime has one exact decoder for
 * every mixed tensor type used by UD-Q4_K_XL.  No re-quantization is involved.
 *
 * The loader first honors FORTAI_GGML_LIBRARY and then checks the machine-local
 * llama.cpp installations used by the benchmark harness.  A build without
 * GGML keeps the existing Q8/F32/F16 path and reports the mixed formats as
 * unavailable rather than silently changing their values.
 */
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef void (*fortai_dequant_fn)(const void *, float *, int64_t);

typedef struct fortai_ggml_context fortai_ggml_context;
typedef struct fortai_ggml_tensor fortai_ggml_tensor;
typedef struct fortai_ggml_cgraph fortai_ggml_cgraph;
typedef struct fortai_ggml_backend fortai_ggml_backend;
typedef struct fortai_ggml_backend_buffer fortai_ggml_backend_buffer;
typedef struct fortai_ggml_backend_buffer_type fortai_ggml_backend_buffer_type;
typedef struct {
    size_t mem_size;
    void * mem_buffer;
    bool no_alloc;
} fortai_ggml_init_params;
typedef fortai_ggml_context * (*fortai_ggml_init_fn)(fortai_ggml_init_params);
typedef void (*fortai_ggml_free_fn)(fortai_ggml_context *);
typedef fortai_ggml_tensor * (*fortai_ggml_new_tensor_2d_fn)(fortai_ggml_context *, int, int64_t, int64_t);
typedef fortai_ggml_tensor * (*fortai_ggml_mul_mat_fn)(fortai_ggml_context *, fortai_ggml_tensor *, fortai_ggml_tensor *);
typedef fortai_ggml_cgraph * (*fortai_ggml_new_graph_fn)(fortai_ggml_context *, size_t, bool);
typedef void (*fortai_ggml_build_fn)(fortai_ggml_cgraph *, fortai_ggml_tensor *);
typedef fortai_ggml_backend * (*fortai_ggml_cpu_init_fn)(void);
typedef void (*fortai_ggml_backend_free_fn)(fortai_ggml_backend *);
typedef fortai_ggml_backend_buffer * (*fortai_ggml_alloc_ctx_fn)(fortai_ggml_context *, fortai_ggml_backend *);
typedef fortai_ggml_backend_buffer * (*fortai_ggml_alloc_buft_fn)(
    fortai_ggml_context *, fortai_ggml_backend_buffer_type *);
typedef fortai_ggml_backend_buffer_type * (*fortai_ggml_repack_buft_fn)(void);
typedef void (*fortai_ggml_buffer_free_fn)(fortai_ggml_backend_buffer *);
typedef void (*fortai_ggml_buffer_usage_fn)(fortai_ggml_backend_buffer *, int);
typedef int (*fortai_ggml_buffer_init_tensor_fn)(fortai_ggml_backend_buffer *, fortai_ggml_tensor *);
typedef void (*fortai_ggml_log_callback)(int, const char *, void *);
typedef void (*fortai_ggml_log_get_fn)(fortai_ggml_log_callback *, void **);
typedef void (*fortai_ggml_log_set_fn)(fortai_ggml_log_callback, void *);
typedef void (*fortai_ggml_tensor_set_fn)(fortai_ggml_tensor *, const void *, size_t, size_t);
typedef void (*fortai_ggml_tensor_get_fn)(const fortai_ggml_tensor *, void *, size_t, size_t);
typedef int (*fortai_ggml_compute_fn)(fortai_ggml_backend *, fortai_ggml_cgraph *);
typedef void (*fortai_ggml_cpu_threads_fn)(fortai_ggml_backend *, int);

typedef struct {
    int type;
    int64_t rows;
    int64_t width;
    const void * weights_host;
    size_t weight_bytes;
    fortai_ggml_context * context;
    fortai_ggml_backend * backend;
    fortai_ggml_backend_buffer * buffer;
    fortai_ggml_backend_buffer * weight_buffer;
    fortai_ggml_context * weight_context;
    fortai_ggml_tensor * weight;
    fortai_ggml_tensor * input;
    fortai_ggml_tensor * output;
    fortai_ggml_cgraph * graph;
} fortai_ggml_cpu_plan;

/* A decode forward repeatedly evaluates several projections with the same
 * activation width.  Keep one GGML graph for those projections instead of
 * entering the backend once per matrix.  This is intentionally limited to
 * the small groups used by Qwen (Q/K/V and FFN gate/up); the scalar API above
 * remains the fallback for arbitrary callers. */
#define FORTAI_GGML_GROUP_MAX 3
typedef struct {
    int count;
    int type[FORTAI_GGML_GROUP_MAX];
    int64_t rows[FORTAI_GGML_GROUP_MAX];
    int64_t width;
    const void *weights_host[FORTAI_GGML_GROUP_MAX];
    size_t weight_bytes[FORTAI_GGML_GROUP_MAX];
    fortai_ggml_context *context;
    fortai_ggml_backend *backend;
    fortai_ggml_backend_buffer *buffer;
    fortai_ggml_backend_buffer *weight_buffer[FORTAI_GGML_GROUP_MAX];
    fortai_ggml_backend_buffer *weight_repack_buffer;
    fortai_ggml_backend_buffer *weight_plain_buffer;
    fortai_ggml_context *weight_context;
    fortai_ggml_tensor *weight[FORTAI_GGML_GROUP_MAX];
    fortai_ggml_tensor *input;
    fortai_ggml_tensor *output[FORTAI_GGML_GROUP_MAX];
    fortai_ggml_cgraph *graph;
} fortai_ggml_cpu_group_plan;

static fortai_ggml_cpu_plan fortai_cpu_plans[256];
static size_t fortai_cpu_plan_count;
static size_t fortai_cpu_plan_bytes;
static fortai_ggml_cpu_group_plan fortai_cpu_group_plans[256];
static size_t fortai_cpu_group_plan_count;
static size_t fortai_cpu_group_plan_bytes;

static void * fortai_ggml_handle;
static int fortai_ggml_attempted;
static void * fortai_ggml_cpu_handle;
static int fortai_ggml_cpu_attempted;
static fortai_ggml_backend *fortai_shared_cpu_backend;
static fortai_ggml_log_callback fortai_previous_log_callback;
static void *fortai_previous_log_user;
static int fortai_log_filter_installed;

static const char * fortai_dequant_name(int type, int64_t *block_width, size_t *block_bytes);

static void fortai_ggml_log_filter(int level, const char *text, void *user) {
    if (text != NULL && strstr(text, "repack tensor") != NULL) return;
    if (fortai_previous_log_callback != NULL)
        fortai_previous_log_callback(level, text, fortai_previous_log_user);
    (void) user;
}

static void fortai_install_log_filter(void *base) {
    if (fortai_log_filter_installed || base == NULL) return;
    fortai_ggml_log_get_fn log_get =
        (fortai_ggml_log_get_fn)dlsym(base, "ggml_log_get");
    fortai_ggml_log_set_fn log_set =
        (fortai_ggml_log_set_fn)dlsym(base, "ggml_log_set");
    if (log_get == NULL || log_set == NULL) return;
    log_get(&fortai_previous_log_callback, &fortai_previous_log_user);
    log_set(fortai_ggml_log_filter, NULL);
    fortai_log_filter_installed = 1;
}

static void * fortai_open_ggml(void) {
    if (fortai_ggml_attempted) return fortai_ggml_handle;
    fortai_ggml_attempted = 1;
    const char * override = getenv("FORTAI_GGML_LIBRARY");
    const char * candidates[] = {
        override,
        "/home/ert/.local/llama.cpp-upstream-main-650913862/lib/libggml-base.so.0",
        "/home/ert/.local/llama.cpp-b10430-cuda/lib/libggml-base.so.0",
        "libggml-base.so.0",
        NULL,
    };
    for (int i = 0; i < 4; ++i) {
        if (candidates[i] == NULL || candidates[i][0] == '\0') continue;
        fortai_ggml_handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (fortai_ggml_handle != NULL) {
            fortai_install_log_filter(fortai_ggml_handle);
            return fortai_ggml_handle;
        }
    }
    return NULL;
}

static void * fortai_open_ggml_cpu(void) {
    if (fortai_ggml_cpu_attempted) return fortai_ggml_cpu_handle;
    fortai_ggml_cpu_attempted = 1;
    const char * override = getenv("FORTAI_GGML_CPU_LIBRARY");
    const char * candidates[] = {
        override,
        "/home/ert/.local/llama.cpp-upstream-main-650913862/lib/libggml-cpu.so.0",
        "/home/ert/.local/llama.cpp-b10430-cuda/lib/libggml-cpu.so.0",
        "libggml-cpu.so.0",
        NULL,
    };
    for (int i = 0; i < 4; ++i) {
        if (candidates[i] == NULL || candidates[i][0] == '\0') continue;
        fortai_ggml_cpu_handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (fortai_ggml_cpu_handle != NULL) return fortai_ggml_cpu_handle;
    }
    return NULL;
}

static fortai_ggml_backend *fortai_get_shared_cpu_backend(void *cpu) {
    if (fortai_shared_cpu_backend != NULL) return fortai_shared_cpu_backend;
    fortai_ggml_cpu_init_fn cpu_init =
        (fortai_ggml_cpu_init_fn)dlsym(cpu, "ggml_backend_cpu_init");
    if (cpu_init == NULL) return NULL;
    fortai_shared_cpu_backend = cpu_init();
    if (fortai_shared_cpu_backend != NULL) {
        fortai_ggml_cpu_threads_fn set_threads =
            (fortai_ggml_cpu_threads_fn)dlsym(cpu, "ggml_backend_cpu_set_n_threads");
        if (set_threads != NULL) {
            const char *text = getenv("OMP_NUM_THREADS");
            int threads = text == NULL ? 0 : atoi(text);
            if (threads > 0) set_threads(fortai_shared_cpu_backend, threads);
        }
    }
    return fortai_shared_cpu_backend;
}

static fortai_ggml_backend_buffer_type *fortai_get_repack_buft(void *cpu) {
    if (cpu == NULL) return NULL;
    /* The CPU extra-buffer entry point is C++ ABI-mangled in current GGML
     * builds.  Keep this optional: older installations simply use the
     * ordinary CPU buffer below. */
    fortai_ggml_repack_buft_fn get_buft =
        (fortai_ggml_repack_buft_fn)dlsym(cpu, "_Z35ggml_backend_cpu_repack_buffer_typev");
    return get_buft == NULL ? NULL : get_buft();
}

static int fortai_repack_supported(int type, int64_t rows) {
    /* The x86 repack backend currently supplies AVX2 kernels for Q4_K and
     * IQ4_NL.  Q6_K remains on the regular vector-dot path on this machine;
     * do not place an unsupported format in CPU_REPACK (its optional trait is
     * null and the setter assumes it is present). */
    return rows > 0 && rows % 8 == 0 && (type == 12 || type == 20);
}

static int fortai_cpu_plan_make(int type, const void * weights, size_t weight_bytes,
    int64_t rows, int64_t width, fortai_ggml_cpu_plan * plan) {
    void * base = fortai_open_ggml();
    void * cpu = fortai_open_ggml_cpu();
    if (base == NULL || cpu == NULL || plan == NULL) return 1;
    fortai_ggml_init_fn init = (fortai_ggml_init_fn)dlsym(base, "ggml_init");
    fortai_ggml_free_fn free_ctx = (fortai_ggml_free_fn)dlsym(base, "ggml_free");
    fortai_ggml_new_tensor_2d_fn new_tensor =
        (fortai_ggml_new_tensor_2d_fn)dlsym(base, "ggml_new_tensor_2d");
    fortai_ggml_mul_mat_fn mul_mat = (fortai_ggml_mul_mat_fn)dlsym(base, "ggml_mul_mat");
    fortai_ggml_new_graph_fn new_graph =
        (fortai_ggml_new_graph_fn)dlsym(base, "ggml_new_graph_custom");
    fortai_ggml_build_fn build = (fortai_ggml_build_fn)dlsym(base, "ggml_build_forward_expand");
    fortai_ggml_alloc_ctx_fn alloc_ctx =
        (fortai_ggml_alloc_ctx_fn)dlsym(base, "ggml_backend_alloc_ctx_tensors");
    fortai_ggml_alloc_buft_fn alloc_buft =
        (fortai_ggml_alloc_buft_fn)dlsym(base, "ggml_backend_alloc_ctx_tensors_from_buft");
    fortai_ggml_tensor_set_fn tensor_set =
        (fortai_ggml_tensor_set_fn)dlsym(base, "ggml_backend_tensor_set");
    if (init == NULL || free_ctx == NULL || new_tensor == NULL || mul_mat == NULL ||
        new_graph == NULL || build == NULL || alloc_ctx == NULL ||
        tensor_set == NULL) return 1;
    fortai_ggml_init_params params = { 2u * 1024u * 1024u, NULL, true };
    plan->context = init(params);
    plan->weight_context = init(params);
    if (plan->context == NULL || plan->weight_context == NULL) return 1;
    plan->weight = new_tensor(plan->weight_context, type, width, rows);
    plan->input = new_tensor(plan->context, 0, width, 1);
    plan->output = plan->weight == NULL || plan->input == NULL ? NULL :
        mul_mat(plan->context, plan->weight, plan->input);
    plan->graph = plan->output == NULL ? NULL : new_graph(plan->context, 4096, false);
    if (plan->graph != NULL) build(plan->graph, plan->output);
    plan->backend = fortai_get_shared_cpu_backend(cpu);
    plan->buffer = plan->backend == NULL ? NULL : alloc_ctx(plan->context, plan->backend);
    fortai_ggml_backend_buffer_type *repack_buft = fortai_get_repack_buft(cpu);
    plan->weight_buffer = repack_buft != NULL && alloc_buft != NULL &&
        fortai_repack_supported(type, rows)
        ? alloc_buft(plan->weight_context, repack_buft) :
        (plan->backend == NULL ? NULL : alloc_ctx(plan->weight_context, plan->backend));
    if (plan->buffer == NULL || plan->weight == NULL || plan->input == NULL ||
        plan->output == NULL || plan->graph == NULL || plan->weight_buffer == NULL) {
        fortai_ggml_buffer_free_fn buffer_free =
            (fortai_ggml_buffer_free_fn)dlsym(base, "ggml_backend_buffer_free");
        if (plan->buffer != NULL) {
            if (buffer_free != NULL) buffer_free(plan->buffer);
        }
        if (plan->weight_buffer != NULL && buffer_free != NULL) buffer_free(plan->weight_buffer);
        free_ctx(plan->context);
        free_ctx(plan->weight_context);
        memset(plan, 0, sizeof(*plan));
        return 1;
    }
    fortai_ggml_buffer_usage_fn set_usage =
        (fortai_ggml_buffer_usage_fn)dlsym(base, "ggml_backend_buffer_set_usage");
    fortai_ggml_buffer_init_tensor_fn init_tensor =
        (fortai_ggml_buffer_init_tensor_fn)dlsym(base, "ggml_backend_buffer_init_tensor");
    if (set_usage != NULL) set_usage(plan->buffer, 1);
    if (set_usage != NULL) set_usage(plan->weight_buffer, 1);
    if (init_tensor != NULL) init_tensor(plan->weight_buffer, plan->weight);
    tensor_set(plan->weight, weights, 0, weight_bytes);
    plan->type = type;
    plan->rows = rows;
    plan->width = width;
    plan->weights_host = weights;
    plan->weight_bytes = weight_bytes;
    return 0;
}

static int fortai_cpu_plan_compute(fortai_ggml_cpu_plan * plan, const float * activation, float * output) {
    void * base = fortai_open_ggml();
    if (base == NULL || plan == NULL || activation == NULL || output == NULL) return 1;
    fortai_ggml_tensor_set_fn tensor_set =
        (fortai_ggml_tensor_set_fn)dlsym(base, "ggml_backend_tensor_set");
    fortai_ggml_tensor_get_fn tensor_get =
        (fortai_ggml_tensor_get_fn)dlsym(base, "ggml_backend_tensor_get");
    fortai_ggml_compute_fn compute =
        (fortai_ggml_compute_fn)dlsym(base, "ggml_backend_graph_compute");
    if (tensor_set == NULL || tensor_get == NULL || compute == NULL) return 1;
    tensor_set(plan->input, activation, 0, (size_t) plan->width * sizeof(float));
    if (compute(plan->backend, plan->graph) != 0) return 1;
    tensor_get(plan->output, output, 0, (size_t) plan->rows * sizeof(float));
    return 0;
}

static int fortai_cpu_plan_cached(int type, const void * weights, size_t weight_bytes,
    int64_t rows, int64_t width, const float * activation, float * output) {
    const char * disabled = getenv("FORTAI_GGML_CPU_CACHE");
    if (disabled != NULL && strcmp(disabled, "0") == 0) return 1;
    for (size_t i = 0; i < fortai_cpu_plan_count; ++i) {
        fortai_ggml_cpu_plan * plan = &fortai_cpu_plans[i];
        if (plan->type == type && plan->weights_host == weights && plan->weight_bytes == weight_bytes &&
            plan->rows == rows && plan->width == width)
            return fortai_cpu_plan_compute(plan, activation, output);
    }
    if (fortai_cpu_plan_count >= sizeof(fortai_cpu_plans) / sizeof(fortai_cpu_plans[0]) ||
        fortai_cpu_plan_bytes + weight_bytes > (size_t) 4u * 1024u * 1024u * 1024u) return 1;
    fortai_ggml_cpu_plan * plan = &fortai_cpu_plans[fortai_cpu_plan_count];
    memset(plan, 0, sizeof(*plan));
    if (fortai_cpu_plan_make(type, weights, weight_bytes, rows, width, plan) != 0) return 1;
    fortai_cpu_plan_count++;
    fortai_cpu_plan_bytes += weight_bytes;
    return fortai_cpu_plan_compute(plan, activation, output);
}

static int fortai_cpu_group_plan_make(int count, const int *types,
    const void * const *weights, const size_t *weight_bytes, const int64_t *rows,
    int64_t width, fortai_ggml_cpu_group_plan *plan) {
    void *base = fortai_open_ggml();
    void *cpu = fortai_open_ggml_cpu();
    if (base == NULL || cpu == NULL || plan == NULL || count < 2 ||
        count > FORTAI_GGML_GROUP_MAX) return 1;
    fortai_ggml_init_fn init = (fortai_ggml_init_fn)dlsym(base, "ggml_init");
    fortai_ggml_free_fn free_ctx = (fortai_ggml_free_fn)dlsym(base, "ggml_free");
    fortai_ggml_new_tensor_2d_fn new_tensor =
        (fortai_ggml_new_tensor_2d_fn)dlsym(base, "ggml_new_tensor_2d");
    fortai_ggml_mul_mat_fn mul_mat = (fortai_ggml_mul_mat_fn)dlsym(base, "ggml_mul_mat");
    fortai_ggml_new_graph_fn new_graph =
        (fortai_ggml_new_graph_fn)dlsym(base, "ggml_new_graph_custom");
    fortai_ggml_build_fn build =
        (fortai_ggml_build_fn)dlsym(base, "ggml_build_forward_expand");
    fortai_ggml_alloc_ctx_fn alloc_ctx =
        (fortai_ggml_alloc_ctx_fn)dlsym(base, "ggml_backend_alloc_ctx_tensors");
    fortai_ggml_alloc_buft_fn alloc_buft =
        (fortai_ggml_alloc_buft_fn)dlsym(base, "ggml_backend_alloc_ctx_tensors_from_buft");
    fortai_ggml_backend_free_fn backend_free =
        (fortai_ggml_backend_free_fn)dlsym(base, "ggml_backend_free");
    fortai_ggml_buffer_free_fn buffer_free =
        (fortai_ggml_buffer_free_fn)dlsym(base, "ggml_backend_buffer_free");
    fortai_ggml_tensor_set_fn tensor_set =
        (fortai_ggml_tensor_set_fn)dlsym(base, "ggml_backend_tensor_set");
    if (init == NULL || free_ctx == NULL || new_tensor == NULL || mul_mat == NULL ||
        new_graph == NULL || build == NULL || backend_free == NULL ||
        alloc_ctx == NULL || buffer_free == NULL || tensor_set == NULL) return 1;

    fortai_ggml_init_params params = { 2u * 1024u * 1024u, NULL, true };
    plan->context = init(params);
    plan->weight_context = init(params);
    if (plan->context == NULL || plan->weight_context == NULL) return 1;
    plan->input = new_tensor(plan->context, 0, width, 1);
    for (int i = 0; i < count; ++i) {
        plan->weight[i] = new_tensor(plan->weight_context, types[i], width, rows[i]);
        plan->output[i] = plan->weight[i] == NULL || plan->input == NULL ? NULL :
            mul_mat(plan->context, plan->weight[i], plan->input);
    }
    plan->graph = new_graph(plan->context, 4096, false);
    if (plan->graph != NULL) {
        for (int i = 0; i < count; ++i)
            if (plan->output[i] != NULL) build(plan->graph, plan->output[i]);
    }
    plan->backend = fortai_get_shared_cpu_backend(cpu);
    plan->buffer = plan->backend == NULL ? NULL : alloc_ctx(plan->context, plan->backend);
    fortai_ggml_backend_buffer_type *repack_buft = fortai_get_repack_buft(cpu);
    if (plan->backend != NULL) {
        int need_repack = 0;
        int need_plain = 0;
        for (int i = 0; i < count; ++i) {
            if (repack_buft != NULL && alloc_buft != NULL && fortai_repack_supported(types[i], rows[i]))
                need_repack = 1;
            else
                need_plain = 1;
        }
        if (need_repack) plan->weight_repack_buffer = alloc_buft(plan->weight_context, repack_buft);
        if (need_plain) plan->weight_plain_buffer = alloc_ctx(plan->weight_context, plan->backend);
        for (int i = 0; i < count; ++i)
            plan->weight_buffer[i] = (repack_buft != NULL && alloc_buft != NULL &&
                fortai_repack_supported(types[i], rows[i]))
                ? plan->weight_repack_buffer : plan->weight_plain_buffer;
    }
    int buffers_ready = plan->backend != NULL;
    if (plan->weight_repack_buffer == NULL && plan->weight_plain_buffer == NULL) buffers_ready = 0;
    for (int i = 0; i < count; ++i) if (plan->weight_buffer[i] == NULL) buffers_ready = 0;
    if (plan->buffer == NULL || !buffers_ready || plan->input == NULL || plan->graph == NULL) {
        if (plan->buffer != NULL) buffer_free(plan->buffer);
        if (plan->weight_repack_buffer != NULL) buffer_free(plan->weight_repack_buffer);
        if (plan->weight_plain_buffer != NULL) buffer_free(plan->weight_plain_buffer);
        free_ctx(plan->context);
        free_ctx(plan->weight_context);
        memset(plan, 0, sizeof(*plan));
        return 1;
    }
    fortai_ggml_buffer_usage_fn set_usage =
        (fortai_ggml_buffer_usage_fn)dlsym(base, "ggml_backend_buffer_set_usage");
    fortai_ggml_buffer_init_tensor_fn init_tensor =
        (fortai_ggml_buffer_init_tensor_fn)dlsym(base, "ggml_backend_buffer_init_tensor");
    if (set_usage != NULL) set_usage(plan->buffer, 1);
    if (set_usage != NULL && plan->weight_repack_buffer != NULL)
        set_usage(plan->weight_repack_buffer, 1);
    if (set_usage != NULL && plan->weight_plain_buffer != NULL)
        set_usage(plan->weight_plain_buffer, 1);
    for (int i = 0; i < count; ++i)
        if (init_tensor != NULL) init_tensor(plan->weight_buffer[i], plan->weight[i]);
    for (int i = 0; i < count; ++i) {
        if (plan->weight[i] == NULL || plan->output[i] == NULL) {
            buffer_free(plan->buffer);
            if (plan->weight_repack_buffer != NULL) buffer_free(plan->weight_repack_buffer);
            if (plan->weight_plain_buffer != NULL) buffer_free(plan->weight_plain_buffer);
            free_ctx(plan->context);
            free_ctx(plan->weight_context);
            memset(plan, 0, sizeof(*plan));
            return 1;
        }
        tensor_set(plan->weight[i], weights[i], 0, weight_bytes[i]);
        plan->type[i] = types[i];
        plan->rows[i] = rows[i];
        plan->weights_host[i] = weights[i];
        plan->weight_bytes[i] = weight_bytes[i];
    }
    plan->count = count;
    plan->width = width;
    return 0;
}

static int fortai_cpu_group_plan_compute(fortai_ggml_cpu_group_plan *plan,
    const float *activation, float * const *outputs) {
    void *base = fortai_open_ggml();
    if (base == NULL || plan == NULL || activation == NULL || outputs == NULL) return 1;
    fortai_ggml_tensor_set_fn tensor_set =
        (fortai_ggml_tensor_set_fn)dlsym(base, "ggml_backend_tensor_set");
    fortai_ggml_tensor_get_fn tensor_get =
        (fortai_ggml_tensor_get_fn)dlsym(base, "ggml_backend_tensor_get");
    fortai_ggml_compute_fn compute =
        (fortai_ggml_compute_fn)dlsym(base, "ggml_backend_graph_compute");
    if (tensor_set == NULL || tensor_get == NULL || compute == NULL) return 1;
    tensor_set(plan->input, activation, 0, (size_t)plan->width * sizeof(float));
    if (compute(plan->backend, plan->graph) != 0) return 1;
    for (int i = 0; i < plan->count; ++i)
        tensor_get(plan->output[i], outputs[i], 0, (size_t)plan->rows[i] * sizeof(float));
    return 0;
}

static int fortai_cpu_group_plan_cached(int count, const int *types,
    const void * const *weights, const size_t *weight_bytes, const int64_t *rows,
    int64_t width, const float *activation, float * const *outputs) {
    const char *disabled = getenv("FORTAI_GGML_CPU_CACHE");
    if (disabled != NULL && strcmp(disabled, "0") == 0) return 1;
    for (size_t p = 0; p < fortai_cpu_group_plan_count; ++p) {
        fortai_ggml_cpu_group_plan *plan = &fortai_cpu_group_plans[p];
        if (plan->count != count || plan->width != width) continue;
        int match = 1;
        for (int i = 0; i < count; ++i) {
            if (plan->type[i] != types[i] || plan->rows[i] != rows[i] ||
                plan->weights_host[i] != weights[i] || plan->weight_bytes[i] != weight_bytes[i]) {
                match = 0;
                break;
            }
        }
        if (match) return fortai_cpu_group_plan_compute(plan, activation, outputs);
    }
    if (fortai_cpu_group_plan_count >=
            sizeof(fortai_cpu_group_plans) / sizeof(fortai_cpu_group_plans[0])) return 1;
    size_t bytes = 0;
    for (int i = 0; i < count; ++i) bytes += weight_bytes[i];
    if (fortai_cpu_group_plan_bytes + bytes > (size_t)4u * 1024u * 1024u * 1024u) return 1;
    fortai_ggml_cpu_group_plan *plan = &fortai_cpu_group_plans[fortai_cpu_group_plan_count];
    memset(plan, 0, sizeof(*plan));
    if (fortai_cpu_group_plan_make(count, types, weights, weight_bytes, rows, width, plan) != 0)
        return 1;
    fortai_cpu_group_plan_count++;
    fortai_cpu_group_plan_bytes += bytes;
    return fortai_cpu_group_plan_compute(plan, activation, outputs);
}

static int fortai_ggml_quant_matvec_group(int count, const int *types,
    const void * const *weights, const size_t *weight_bytes, const int64_t *rows,
    int64_t width, const float *activation, float * const *outputs) {
    const char *disabled = getenv("FORTAI_GGML_CPU_GROUP");
    if (disabled != NULL && strcmp(disabled, "0") == 0) return 1;
    if (count < 2 || count > FORTAI_GGML_GROUP_MAX || types == NULL || weights == NULL ||
        weight_bytes == NULL || rows == NULL || activation == NULL || outputs == NULL ||
        width <= 0) return 1;
    for (int i = 0; i < count; ++i) {
        int64_t block_width = 0;
        size_t block_bytes = 0;
        if (fortai_dequant_name(types[i], &block_width, &block_bytes) == NULL ||
            weights[i] == NULL || outputs[i] == NULL || rows[i] <= 0 ||
            width % block_width != 0 ||
            (size_t)(width / block_width) * block_bytes * (size_t)rows[i] != weight_bytes[i]) return 1;
    }
    return fortai_cpu_group_plan_cached(count, types, weights, weight_bytes, rows,
        width, activation, outputs);
}

int fortai_ggml_quant_matvec_pair(int first_type, const void *first_weights,
    size_t first_weight_bytes, int64_t first_rows, int second_type, const void *second_weights,
    size_t second_weight_bytes, int64_t second_rows, int64_t width, const float *activation,
    float *first_output, float *second_output) {
    const int types[2] = {first_type, second_type};
    const void *weights[2] = {first_weights, second_weights};
    const size_t weight_bytes[2] = {first_weight_bytes, second_weight_bytes};
    const int64_t rows[2] = {first_rows, second_rows};
    float *outputs[2] = {first_output, second_output};
    return fortai_ggml_quant_matvec_group(2, types, weights, weight_bytes, rows,
        width, activation, outputs);
}

int fortai_ggml_quant_matvec_triplet(int first_type, const void *first_weights,
    size_t first_weight_bytes, int64_t first_rows, int second_type, const void *second_weights,
    size_t second_weight_bytes, int64_t second_rows, int third_type, const void *third_weights,
    size_t third_weight_bytes, int64_t third_rows, int64_t width, const float *activation,
    float *first_output, float *second_output, float *third_output) {
    const int types[3] = {first_type, second_type, third_type};
    const void *weights[3] = {first_weights, second_weights, third_weights};
    const size_t weight_bytes[3] = {first_weight_bytes, second_weight_bytes, third_weight_bytes};
    const int64_t rows[3] = {first_rows, second_rows, third_rows};
    float *outputs[3] = {first_output, second_output, third_output};
    return fortai_ggml_quant_matvec_group(3, types, weights, weight_bytes, rows,
        width, activation, outputs);
}

void fortai_ggml_quant_cache_clear(void) {
    if (fortai_cpu_plan_count == 0 && fortai_cpu_group_plan_count == 0 &&
        fortai_shared_cpu_backend == NULL) return;
    void * base = fortai_open_ggml();
    if (base == NULL) return;
    fortai_ggml_buffer_free_fn buffer_free =
        (fortai_ggml_buffer_free_fn)dlsym(base, "ggml_backend_buffer_free");
    fortai_ggml_backend_free_fn backend_free =
        (fortai_ggml_backend_free_fn)dlsym(base, "ggml_backend_free");
    fortai_ggml_free_fn free_ctx = (fortai_ggml_free_fn)dlsym(base, "ggml_free");
    for (size_t i = 0; i < fortai_cpu_plan_count; ++i) {
        if (buffer_free != NULL && fortai_cpu_plans[i].buffer != NULL)
            buffer_free(fortai_cpu_plans[i].buffer);
        if (buffer_free != NULL && fortai_cpu_plans[i].weight_buffer != NULL)
            buffer_free(fortai_cpu_plans[i].weight_buffer);
        if (free_ctx != NULL && fortai_cpu_plans[i].context != NULL)
            free_ctx(fortai_cpu_plans[i].context);
        if (free_ctx != NULL && fortai_cpu_plans[i].weight_context != NULL)
            free_ctx(fortai_cpu_plans[i].weight_context);
    }
    for (size_t i = 0; i < fortai_cpu_group_plan_count; ++i) {
        if (buffer_free != NULL && fortai_cpu_group_plans[i].buffer != NULL)
            buffer_free(fortai_cpu_group_plans[i].buffer);
        if (buffer_free != NULL) {
            if (fortai_cpu_group_plans[i].weight_repack_buffer != NULL)
                buffer_free(fortai_cpu_group_plans[i].weight_repack_buffer);
            if (fortai_cpu_group_plans[i].weight_plain_buffer != NULL)
                buffer_free(fortai_cpu_group_plans[i].weight_plain_buffer);
        }
        if (free_ctx != NULL && fortai_cpu_group_plans[i].context != NULL)
            free_ctx(fortai_cpu_group_plans[i].context);
        if (free_ctx != NULL && fortai_cpu_group_plans[i].weight_context != NULL)
            free_ctx(fortai_cpu_group_plans[i].weight_context);
    }
    if (backend_free != NULL && fortai_shared_cpu_backend != NULL)
        backend_free(fortai_shared_cpu_backend);
    fortai_shared_cpu_backend = NULL;
    memset(fortai_cpu_plans, 0, sizeof(fortai_cpu_plans));
    fortai_cpu_plan_count = 0;
    fortai_cpu_plan_bytes = 0;
    memset(fortai_cpu_group_plans, 0, sizeof(fortai_cpu_group_plans));
    fortai_cpu_group_plan_count = 0;
    fortai_cpu_group_plan_bytes = 0;
}

static const char * fortai_dequant_name(int type, int64_t * block_width, size_t * block_bytes) {
    switch (type) {
    case 11: *block_width = 256; *block_bytes = 110; return "dequantize_row_q3_K";
    case 12: *block_width = 256; *block_bytes = 144; return "dequantize_row_q4_K";
    case 13: *block_width = 256; *block_bytes = 176; return "dequantize_row_q5_K";
    case 14: *block_width = 256; *block_bytes = 210; return "dequantize_row_q6_K";
    case 20: *block_width = 32;  *block_bytes = 18;  return "dequantize_row_iq4_nl";
    case 21: *block_width = 256; *block_bytes = 110; return "dequantize_row_iq3_s";
    case 23: *block_width = 256; *block_bytes = 136; return "dequantize_row_iq4_xs";
    default: return NULL;
    }
}

static fortai_dequant_fn fortai_resolve(int type, int64_t * block_width, size_t * block_bytes) {
    const char * name = fortai_dequant_name(type, block_width, block_bytes);
    if (name == NULL) return NULL;
    void * handle = fortai_open_ggml();
    if (handle == NULL) return NULL;
    return (fortai_dequant_fn)dlsym(handle, name);
}

int fortai_ggml_quant_available(int type) {
    int64_t block_width = 0;
    size_t block_bytes = 0;
    return fortai_resolve(type, &block_width, &block_bytes) != NULL;
}

int fortai_ggml_dequantize_row(int type, const void * input, float * output, int64_t width) {
    int64_t block_width = 0;
    size_t block_bytes = 0;
    fortai_dequant_fn fn = fortai_resolve(type, &block_width, &block_bytes);
    if (fn == NULL || input == NULL || output == NULL || width <= 0 || width % block_width != 0) return 1;
    fn(input, output, width);
    return 0;
}

/* Execute a complete mixed-quantized matrix-vector product through the same
 * GGML CPU backend used by llama.cpp.  This keeps activation quantization,
 * SIMD dot kernels, and thread scheduling on the optimized implementation;
 * the scalar row decoder remains available for row access and as a fallback. */
int fortai_ggml_quant_matvec(int type, const void * weights, size_t weight_bytes,
    int64_t rows, int64_t width, const float * activation, float * output) {
    int64_t block_width = 0;
    size_t block_bytes = 0;
    if (fortai_dequant_name(type, &block_width, &block_bytes) == NULL ||
        weights == NULL || activation == NULL || output == NULL || rows <= 0 ||
        width <= 0 || width % block_width != 0) return 1;
    if ((size_t) (width / block_width) * block_bytes * (size_t) rows != weight_bytes) return 1;
    if (fortai_cpu_plan_cached(type, weights, weight_bytes, rows, width, activation, output) == 0)
        return 0;

    void * base = fortai_open_ggml();
    void * cpu = fortai_open_ggml_cpu();
    if (base == NULL || cpu == NULL) return 1;

    fortai_ggml_init_fn init = (fortai_ggml_init_fn)dlsym(base, "ggml_init");
    fortai_ggml_free_fn free_ctx = (fortai_ggml_free_fn)dlsym(base, "ggml_free");
    fortai_ggml_new_tensor_2d_fn new_tensor =
        (fortai_ggml_new_tensor_2d_fn)dlsym(base, "ggml_new_tensor_2d");
    fortai_ggml_mul_mat_fn mul_mat = (fortai_ggml_mul_mat_fn)dlsym(base, "ggml_mul_mat");
    fortai_ggml_new_graph_fn new_graph =
        (fortai_ggml_new_graph_fn)dlsym(base, "ggml_new_graph_custom");
    fortai_ggml_build_fn build = (fortai_ggml_build_fn)dlsym(base, "ggml_build_forward_expand");
    fortai_ggml_cpu_init_fn cpu_init =
        (fortai_ggml_cpu_init_fn)dlsym(cpu, "ggml_backend_cpu_init");
    fortai_ggml_backend_free_fn backend_free =
        (fortai_ggml_backend_free_fn)dlsym(base, "ggml_backend_free");
    fortai_ggml_alloc_ctx_fn alloc_ctx =
        (fortai_ggml_alloc_ctx_fn)dlsym(base, "ggml_backend_alloc_ctx_tensors");
    fortai_ggml_buffer_free_fn buffer_free =
        (fortai_ggml_buffer_free_fn)dlsym(base, "ggml_backend_buffer_free");
    fortai_ggml_tensor_set_fn tensor_set =
        (fortai_ggml_tensor_set_fn)dlsym(base, "ggml_backend_tensor_set");
    fortai_ggml_tensor_get_fn tensor_get =
        (fortai_ggml_tensor_get_fn)dlsym(base, "ggml_backend_tensor_get");
    fortai_ggml_compute_fn compute =
        (fortai_ggml_compute_fn)dlsym(base, "ggml_backend_graph_compute");
    fortai_ggml_cpu_threads_fn set_threads =
        (fortai_ggml_cpu_threads_fn)dlsym(cpu, "ggml_backend_cpu_set_n_threads");
    if (init == NULL || free_ctx == NULL || new_tensor == NULL || mul_mat == NULL ||
        new_graph == NULL || build == NULL || cpu_init == NULL || backend_free == NULL ||
        alloc_ctx == NULL || buffer_free == NULL || tensor_set == NULL || tensor_get == NULL ||
        compute == NULL) return 1;

    fortai_ggml_init_params params = {
        2u * 1024u * 1024u, NULL, true
    };
    fortai_ggml_context * context = init(params);
    if (context == NULL) return 1;
    fortai_ggml_tensor * weight = new_tensor(context, type, width, rows);
    fortai_ggml_tensor * input = new_tensor(context, 0, width, 1);
    fortai_ggml_tensor * result = weight == NULL || input == NULL ? NULL : mul_mat(context, weight, input);
    fortai_ggml_cgraph * graph = result == NULL ? NULL : new_graph(context, 4096, false);
    if (graph != NULL) build(graph, result);
    fortai_ggml_backend * backend = cpu_init();
    fortai_ggml_backend_buffer * buffer = backend == NULL ? NULL : alloc_ctx(context, backend);
    int code = 1;
    if (buffer != NULL && weight != NULL && input != NULL && result != NULL && graph != NULL) {
        tensor_set(weight, weights, 0, weight_bytes);
        tensor_set(input, activation, 0, (size_t) width * sizeof(float));
        if (set_threads != NULL) {
            const char * text = getenv("OMP_NUM_THREADS");
            int threads = text == NULL ? 0 : atoi(text);
            if (threads > 0) set_threads(backend, threads);
        }
        if (compute(backend, graph) == 0) {
            tensor_get(result, output, 0, (size_t) rows * sizeof(float));
            code = 0;
        }
    }
    if (buffer != NULL) buffer_free(buffer);
    if (backend != NULL) backend_free(backend);
    free_ctx(context);
    return code;
}

float fortai_ggml_quant_dot(int type, const void * input, const float * vector, int64_t width) {
    int64_t block_width = 0;
    size_t block_bytes = 0;
    fortai_dequant_fn fn = fortai_resolve(type, &block_width, &block_bytes);
    if (fn == NULL || input == NULL || vector == NULL || width <= 0 || width % block_width != 0) return 0.0f;

    float values[256];
    float result = 0.0f;
    const unsigned char * bytes = (const unsigned char *)input;
    for (int64_t offset = 0; offset < width; offset += block_width) {
        fn(bytes, values, block_width);
        for (int64_t i = 0; i < block_width; ++i) result += values[i] * vector[offset + i];
        bytes += block_bytes;
    }
    return result;
}
