/*
 * Small runtime adapter for the public llama.cpp C API.
 *
 * FortAI's native Qwen3.5 implementation remains the fallback.  This adapter
 * is intentionally late-bound so a normal build does not require llama.cpp
 * headers or link against a particular release; the installed library is
 * selected at runtime and its graph/CPU-repack/CUDA backends stay resident.
 */
#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

typedef struct fortai_llama_model fortai_llama_model;
typedef struct fortai_llama_context fortai_llama_context;
typedef struct fortai_llama_vocab fortai_llama_vocab;
typedef struct fortai_llama_batch {
    int32_t n_tokens;
    int32_t *token;
    float *embd;
    int32_t *pos;
    int32_t *n_seq_id;
    int32_t **seq_id;
    int8_t *logits;
} fortai_llama_batch;

typedef struct {
    void *devices;
    const void *tensor_buft_overrides;
    int32_t n_gpu_layers;
    int32_t split_mode;
    int32_t load_mode;
    int32_t main_gpu;
    const float *tensor_split;
    void *progress_callback;
    void *progress_callback_user_data;
    const void *kv_overrides;
    bool vocab_only;
    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;
    bool load_mtp;
} fortai_llama_model_params;

typedef struct {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;
    uint32_t n_outputs_max;
    uint32_t n_outputs_max_per_seq;
    int32_t n_threads;
    int32_t n_threads_batch;
    int32_t ctx_type;
    int32_t rope_scaling_type;
    int32_t pooling_type;
    int32_t attention_type;
    int32_t flash_attn_type;
    float rope_freq_base;
    float rope_freq_scale;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float defrag_thold;
    void *cb_eval;
    void *cb_eval_user_data;
    int32_t type_k;
    int32_t type_v;
    void *abort_callback;
    void *abort_callback_data;
    bool embeddings;
    bool offload_kqv;
    bool no_perf;
    bool op_offload;
    bool swa_full;
    bool kv_unified;
    void *samplers;
    size_t n_samplers;
    fortai_llama_context *ctx_other;
} fortai_llama_context_params;

typedef fortai_llama_model_params (*fortai_model_default_fn)(void);
typedef fortai_llama_context_params (*fortai_context_default_fn)(void);
typedef void (*fortai_backend_init_fn)(void);
typedef void (*fortai_backend_free_fn)(void);
typedef fortai_llama_model *(*fortai_model_load_fn)(
    const char *, fortai_llama_model_params);
typedef fortai_llama_context *(*fortai_context_init_fn)(
    const fortai_llama_model *, fortai_llama_context_params);
typedef const fortai_llama_vocab *(*fortai_model_vocab_fn)(const fortai_llama_model *);
typedef int32_t (*fortai_vocab_count_fn)(const fortai_llama_vocab *);
typedef int32_t (*fortai_model_layer_count_fn)(const fortai_llama_model *);
typedef int32_t (*fortai_model_layer_nextn_count_fn)(const fortai_llama_model *);
typedef int32_t (*fortai_model_embd_out_fn)(const fortai_llama_model *);
typedef void (*fortai_model_free_fn)(fortai_llama_model *);
typedef void (*fortai_context_free_fn)(fortai_llama_context *);
typedef void (*fortai_set_threads_fn)(fortai_llama_context *, int32_t, int32_t);
typedef fortai_llama_batch (*fortai_batch_init_fn)(int32_t, int32_t, int32_t);
typedef void (*fortai_batch_free_fn)(fortai_llama_batch);
typedef int32_t (*fortai_decode_fn)(fortai_llama_context *, fortai_llama_batch);
typedef float *(*fortai_logits_fn)(fortai_llama_context *, int32_t);
typedef void *(*fortai_memory_fn)(const fortai_llama_context *);
typedef void (*fortai_memory_clear_fn)(void *, bool);
typedef bool (*fortai_memory_seq_rm_fn)(void *, int32_t, int32_t, int32_t);
typedef void (*fortai_set_embeddings_nextn_fn)(fortai_llama_context *, bool, bool);
typedef void (*fortai_set_nextn_layer_offset_fn)(fortai_llama_context *, int32_t);
typedef float *(*fortai_get_embeddings_nextn_fn)(fortai_llama_context *);
typedef float *(*fortai_get_embeddings_nextn_ith_fn)(fortai_llama_context *, int32_t);
typedef void (*fortai_synchronize_fn)(fortai_llama_context *);
typedef void (*fortai_log_callback_fn)(int, const char *, void *);
typedef void (*fortai_log_set_fn)(fortai_log_callback_fn, void *);
typedef struct {
    bool cpumask[512];
    int32_t n_threads;
    int32_t prio;
    uint32_t poll;
    bool strict_cpu;
    bool paused;
} fortai_threadpool_params;
typedef struct fortai_threadpool fortai_threadpool;
typedef void (*fortai_threadpool_params_init_fn)(fortai_threadpool_params *, int32_t);
typedef fortai_threadpool *(*fortai_threadpool_new_fn)(fortai_threadpool_params *);
typedef void (*fortai_threadpool_free_fn)(fortai_threadpool *);
typedef void (*fortai_attach_threadpool_fn)(fortai_llama_context *, fortai_threadpool *,
    fortai_threadpool *);

typedef struct {
    void *library;
    fortai_backend_init_fn backend_init;
    fortai_backend_free_fn backend_free;
    fortai_model_default_fn model_default;
    fortai_context_default_fn context_default;
    fortai_model_load_fn model_load;
    fortai_context_init_fn context_init;
    fortai_model_vocab_fn model_vocab;
    fortai_vocab_count_fn vocab_count;
    fortai_model_layer_count_fn layer_count;
    fortai_model_layer_nextn_count_fn layer_nextn_count;
    fortai_model_embd_out_fn embd_out;
    fortai_model_free_fn model_free;
    fortai_context_free_fn context_free;
    fortai_set_threads_fn set_threads;
    fortai_batch_init_fn batch_init;
    fortai_batch_free_fn batch_free;
    fortai_decode_fn decode;
    fortai_logits_fn logits;
    fortai_memory_fn memory;
    fortai_memory_clear_fn memory_clear;
    fortai_memory_seq_rm_fn memory_seq_rm;
    fortai_set_embeddings_nextn_fn set_embeddings_nextn;
    fortai_set_nextn_layer_offset_fn set_nextn_layer_offset;
    fortai_get_embeddings_nextn_fn get_embeddings_nextn;
    fortai_get_embeddings_nextn_ith_fn get_embeddings_nextn_ith;
    fortai_synchronize_fn synchronize;
    fortai_log_set_fn log_set;
    fortai_attach_threadpool_fn attach_threadpool;
} fortai_llama_api;

typedef struct {
    fortai_llama_api api;
    fortai_llama_model *model;
    fortai_llama_context *context;
    fortai_llama_batch batch;
    int vocab;
    int max_context;
    char *path;
    fortai_threadpool *threadpool;
    fortai_threadpool_free_fn threadpool_free;
    fortai_llama_model *draft_model;
    fortai_llama_context *draft_context;
    fortai_llama_batch draft_batch;
    int draft_vocab;
    int spec_max;
    int spec_enabled;
    fortai_llama_context *mtp_context;
    fortai_llama_batch mtp_batch;
    int mtp_enabled;
    int mtp_embd;
} fortai_llama_handle;

static fortai_llama_api fortai_api;
static int fortai_api_attempted;
static int fortai_backend_initialized;

static void fortai_fast_diag(const char *message) {
    if (getenv("FORTAI_LLAMA_FAST_VERBOSE") != NULL)
        fprintf(stderr, "fortai-fast: %s\n", message);
}

/* The model runner is also used underneath the local llama.cpp-compatible
 * service.  Keep the fast adapter late-bound, but honor the same small set of
 * model/context controls through environment variables.  The LLAMACPP_* and
 * LLAMA_ARG_* aliases make the service profile usable without a second config
 * translation layer; FORTAI_* remains the explicit per-process override. */
static const char *fortai_env3(const char *fortai_name, const char *llama_name,
    const char *service_name) {
    const char *value = getenv(fortai_name);
    if (value != NULL && value[0] != '\0') return value;
    value = getenv(llama_name);
    if (value != NULL && value[0] != '\0') return value;
    value = getenv(service_name);
    if (value != NULL && value[0] != '\0') return value;
    return NULL;
}

static int fortai_parse_bool(const char *value, int fallback) {
    if (value == NULL || value[0] == '\0') return fallback;
    if (strcasecmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
        strcasecmp(value, "on") == 0 || strcasecmp(value, "yes") == 0)
        return 1;
    if (strcasecmp(value, "0") == 0 || strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "off") == 0 || strcasecmp(value, "no") == 0)
        return 0;
    return fallback;
}

static int fortai_parse_int(const char *value, int fallback, int minimum, int maximum) {
    char *end = NULL;
    long parsed;
    if (value == NULL || value[0] == '\0') return fallback;
    parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minimum || parsed > maximum)
        return fallback;
    return (int)parsed;
}

static int fortai_parse_flash_attn(const char *value, int fallback) {
    if (value == NULL || value[0] == '\0') return fallback;
    if (strcasecmp(value, "auto") == 0) return -1;
    if (strcasecmp(value, "on") == 0 || strcasecmp(value, "true") == 0)
        return 1;
    if (strcasecmp(value, "off") == 0 || strcasecmp(value, "false") == 0)
        return 0;
    return fallback;
}

static int fortai_parse_cache_type(const char *value, int fallback) {
    if (value == NULL || value[0] == '\0') return fallback;
    if (strcasecmp(value, "f32") == 0) return 0;
    if (strcasecmp(value, "f16") == 0) return 1;
    if (strcasecmp(value, "q4_0") == 0) return 2;
    if (strcasecmp(value, "q4_1") == 0) return 3;
    if (strcasecmp(value, "q5_0") == 0) return 6;
    if (strcasecmp(value, "q5_1") == 0) return 7;
    if (strcasecmp(value, "q8_0") == 0) return 8;
    return fallback;
}

static int fortai_path_requests_mtp(const char *draft_path) {
    const char *value = fortai_env3("FORTAI_SPEC_TYPE", "LLAMA_ARG_SPEC_TYPE",
        "LLAMACPP_SPEC_TYPE");
    if (value != NULL && strcasecmp(value, "draft-mtp") == 0) return 1;
    if (draft_path == NULL) return 0;
    /* The published Qwen3.8 head is a tiny GGUF containing only blk.*.nextn
     * tensors.  Treat that naming convention as an MTP request instead of
     * trying to initialize it as a standalone transformer (which is invalid
     * and can crash older llama.cpp graph builders). */
    return strcasestr(draft_path, "mtp") != NULL;
}

static const float *fortai_parse_tensor_split(const char *value) {
    static float split[16];
    char *end;
    int count = 0;
    if (value == NULL || value[0] == '\0') return NULL;
    while (*value != '\0' && count < (int)(sizeof(split) / sizeof(split[0]))) {
        float ratio = strtof(value, &end);
        if (end == value || ratio <= 0.0f) return NULL;
        split[count++] = ratio;
        if (*end == '\0') break;
        if (*end != ',') return NULL;
        value = end + 1;
    }
    if (count == 0 || *value != '\0') return NULL;
    return split;
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
static __attribute__((target("avx2"))) int fortai_argmax_avx2(const float *values, int count,
    float *sum_out) {
    const __m256 negative_infinity = _mm256_set1_ps(-INFINITY);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i index_step = _mm256_set1_epi32(8);
    __m256 maximum = negative_infinity;
    __m256 sum = _mm256_setzero_ps();
    __m256i indices = zero;
    __m256i next_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    int offset = 0;
    for (; offset + 8 <= count; offset += 8) {
        const __m256 current = _mm256_loadu_ps(values + offset);
        sum = _mm256_add_ps(sum, current);
        const __m256 greater = _mm256_cmp_ps(current, maximum, _CMP_GT_OQ);
        maximum = _mm256_blendv_ps(maximum, current, greater);
        indices = _mm256_blendv_epi8(indices, next_indices, _mm256_castps_si256(greater));
        next_indices = _mm256_add_epi32(next_indices, index_step);
    }
    float maxima[8];
    int best_indices[8];
    _mm256_storeu_ps(maxima, maximum);
    _mm256_storeu_si256((__m256i *)best_indices, indices);
    int best = best_indices[0];
    float best_value = maxima[0];
    for (int lane = 1; lane < 8; ++lane) {
        if (maxima[lane] > best_value ||
            (maxima[lane] == best_value && best_indices[lane] < best)) {
            best_value = maxima[lane];
            best = best_indices[lane];
        }
    }
    float total = 0.0f;
    float sums[8];
    _mm256_storeu_ps(sums, sum);
    for (int lane = 0; lane < 8; ++lane) total += sums[lane];
    for (; offset < count; ++offset) {
        total += values[offset];
        if (values[offset] > best_value) {
            best_value = values[offset];
            best = offset;
        }
    }
    *sum_out = total;
    return best;
}
#endif

static int fortai_argmax_f32(const float *values, int count, float *sum_out) {
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
    static int use_avx2 = -1;
    if (use_avx2 < 0) use_avx2 = __builtin_cpu_supports("avx2") != 0;
    if (use_avx2) return fortai_argmax_avx2(values, count, sum_out);
#endif
    int best = 0;
    float best_value = values[0];
    float total = values[0];
    for (int i = 1; i < count; ++i) {
        total += values[i];
        if (values[i] > best_value) {
            best_value = values[i];
            best = i;
        }
    }
    *sum_out = total;
    return best;
}

static void fortai_quiet_log(int level, const char *text, void *user) {
    (void)level;
    (void)text;
    (void)user;
}

static void *fortai_symbol(void *library, const char *name) {
    return library == NULL ? NULL : dlsym(library, name);
}

static void fortai_load_dependencies(const char *directory) {
    const char *names[] = {
        "libggml.so.0", "libggml-base.so.0", "libggml-cpu.so.0",
        "libggml-cuda.so.0"
    };
    char path[1024];
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        int n = snprintf(path, sizeof(path), "%s/%s", directory, names[i]);
        if (n > 0 && (size_t)n < sizeof(path))
            (void)dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    }
}

static void *fortai_open_llama(void) {
    if (fortai_api_attempted) return fortai_api.library;
    fortai_api_attempted = 1;
    const char *override = getenv("FORTAI_LLAMA_FAST_LIBRARY");
    const char *directory = getenv("FORTAI_LLAMA_FAST_LIBRARY_DIR");
    const char *candidates[] = {
        override,
        "/home/ert/.local/llama.cpp-b10566-cuda/libllama.so.0",
        "/home/ert/.local/llama.cpp-upstream-main-650913862/libllama.so.0",
        "libllama.so.0",
        NULL
    };
    for (int i = 0; i < 4; ++i) {
        if (candidates[i] == NULL || candidates[i][0] == '\0') continue;
        char local_path[1024];
        const char *path = candidates[i];
        if (i == 0 && directory != NULL && directory[0] != '\0' &&
            strchr(path, '/') == NULL) {
            int n = snprintf(local_path, sizeof(local_path), "%s/%s", directory, path);
            if (n <= 0 || (size_t)n >= sizeof(local_path)) continue;
            path = local_path;
        }
        char local_directory[1024];
        const char *slash = strrchr(path, '/');
        if (slash != NULL) {
            size_t length = (size_t)(slash - path);
            if (length < sizeof(local_directory)) {
                memcpy(local_directory, path, length);
                local_directory[length] = '\0';
                fortai_load_dependencies(local_directory);
            }
        }
        fortai_api.library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (fortai_api.library != NULL) break;
    }
    if (fortai_api.library == NULL) return NULL;
#define FORTAI_LLAMA_LOAD(field, symbol) \
    fortai_api.field = (typeof(fortai_api.field))fortai_symbol(fortai_api.library, symbol)
    FORTAI_LLAMA_LOAD(backend_init, "llama_backend_init");
    FORTAI_LLAMA_LOAD(backend_free, "llama_backend_free");
    FORTAI_LLAMA_LOAD(model_default, "llama_model_default_params");
    FORTAI_LLAMA_LOAD(context_default, "llama_context_default_params");
    FORTAI_LLAMA_LOAD(model_load, "llama_model_load_from_file");
    FORTAI_LLAMA_LOAD(context_init, "llama_init_from_model");
    FORTAI_LLAMA_LOAD(model_vocab, "llama_model_get_vocab");
    FORTAI_LLAMA_LOAD(vocab_count, "llama_vocab_n_tokens");
    FORTAI_LLAMA_LOAD(layer_count, "llama_model_n_layer");
    FORTAI_LLAMA_LOAD(layer_nextn_count, "llama_model_n_layer_nextn");
    FORTAI_LLAMA_LOAD(embd_out, "llama_model_n_embd_out");
    FORTAI_LLAMA_LOAD(model_free, "llama_model_free");
    FORTAI_LLAMA_LOAD(context_free, "llama_free");
    FORTAI_LLAMA_LOAD(set_threads, "llama_set_n_threads");
    FORTAI_LLAMA_LOAD(batch_init, "llama_batch_init");
    FORTAI_LLAMA_LOAD(batch_free, "llama_batch_free");
    FORTAI_LLAMA_LOAD(decode, "llama_decode");
    FORTAI_LLAMA_LOAD(logits, "llama_get_logits_ith");
    FORTAI_LLAMA_LOAD(memory, "llama_get_memory");
    FORTAI_LLAMA_LOAD(memory_clear, "llama_memory_clear");
    FORTAI_LLAMA_LOAD(memory_seq_rm, "llama_memory_seq_rm");
    FORTAI_LLAMA_LOAD(set_embeddings_nextn, "llama_set_embeddings_nextn");
    FORTAI_LLAMA_LOAD(set_nextn_layer_offset, "llama_set_nextn_layer_offset");
    FORTAI_LLAMA_LOAD(get_embeddings_nextn, "llama_get_embeddings_nextn");
    FORTAI_LLAMA_LOAD(get_embeddings_nextn_ith, "llama_get_embeddings_nextn_ith");
    /* llama-ext.h is intentionally a C++ staging header in current builds,
     * so these four symbols are exported with their Itanium names. */
    if (fortai_api.set_embeddings_nextn == NULL)
        fortai_api.set_embeddings_nextn = (fortai_set_embeddings_nextn_fn)
            fortai_symbol(fortai_api.library, "_Z26llama_set_embeddings_nextnP13llama_contextbb");
    if (fortai_api.set_nextn_layer_offset == NULL)
        fortai_api.set_nextn_layer_offset = (fortai_set_nextn_layer_offset_fn)
            fortai_symbol(fortai_api.library, "_Z28llama_set_nextn_layer_offsetP13llama_contexti");
    if (fortai_api.get_embeddings_nextn == NULL)
        fortai_api.get_embeddings_nextn = (fortai_get_embeddings_nextn_fn)
            fortai_symbol(fortai_api.library, "_Z26llama_get_embeddings_nextnP13llama_context");
    if (fortai_api.get_embeddings_nextn_ith == NULL)
        fortai_api.get_embeddings_nextn_ith = (fortai_get_embeddings_nextn_ith_fn)
            fortai_symbol(fortai_api.library, "_Z30llama_get_embeddings_nextn_ithP13llama_contexti");
    FORTAI_LLAMA_LOAD(synchronize, "llama_synchronize");
    FORTAI_LLAMA_LOAD(log_set, "llama_log_set");
    FORTAI_LLAMA_LOAD(attach_threadpool, "llama_attach_threadpool");
#undef FORTAI_LLAMA_LOAD
    if (fortai_api.model_default == NULL || fortai_api.context_default == NULL ||
        fortai_api.model_load == NULL || fortai_api.context_init == NULL ||
        fortai_api.model_vocab == NULL || fortai_api.vocab_count == NULL ||
        fortai_api.model_free == NULL || fortai_api.context_free == NULL ||
        fortai_api.set_threads == NULL || fortai_api.batch_init == NULL ||
        fortai_api.batch_free == NULL || fortai_api.decode == NULL ||
        fortai_api.logits == NULL)
        return NULL;
    if (fortai_api.backend_init != NULL && !fortai_backend_initialized) {
        fortai_api.backend_init();
        fortai_backend_initialized = 1;
    }
    if (fortai_api.log_set != NULL && getenv("FORTAI_LLAMA_FAST_VERBOSE") == NULL)
        fortai_api.log_set(fortai_quiet_log, NULL);
    return fortai_api.library;
}

int fortai_llama_fast_available(void) {
    return fortai_open_llama() == NULL ? 0 : 1;
}

int fortai_llama_fast_context_destroy(void *opaque);

static void fortai_warm_context(fortai_llama_handle *handle,
    fortai_llama_context *context, fortai_llama_batch *batch) {
    const char *warmup = getenv("FORTAI_LLAMA_FAST_WARMUP");
    if (warmup != NULL && strcmp(warmup, "0") == 0) return;
    batch->n_tokens = 1;
    batch->token[0] = 0;
    batch->pos[0] = 0;
    batch->n_seq_id[0] = 1;
    batch->seq_id[0][0] = 0;
    batch->logits[0] = 1;
    (void)handle->api.decode(context, *batch);
    if (handle->api.synchronize != NULL) handle->api.synchronize(context);
    if (handle->api.memory != NULL && handle->api.memory_clear != NULL)
        handle->api.memory_clear(handle->api.memory(context), true);
}

int fortai_llama_fast_context_create(const char *path, int context_size, int threads,
    int gpu_layers, int main_gpu, void **out, int *vocab, int *layers) {
    if (path == NULL || out == NULL || context_size <= 0 || threads <= 0)
        return 1;
    *out = NULL;
    void *library = fortai_open_llama();
    if (library == NULL) return 2;
    fortai_llama_handle *handle = (fortai_llama_handle *)calloc(1, sizeof(*handle));
    if (handle == NULL) return 2;
    handle->api = fortai_api;
    fortai_llama_model_params model_params = handle->api.model_default();
    model_params.n_gpu_layers = gpu_layers;
    model_params.main_gpu = main_gpu;
    model_params.use_extra_bufts = true;
    /* Loading the optional NextN/MTP tensors is metadata-gated inside
     * llama.cpp and does not change ordinary-model graphs.  Keep it enabled
     * by default so a Qwen3.8-style single-file MTP head is available to the
     * resident context; callers can disable it for strict legacy parity. */
    model_params.load_mtp = fortai_parse_bool(fortai_env3(
        "FORTAI_LOAD_MTP", "LLAMA_ARG_LOAD_MTP", "LLAMACPP_LOAD_MTP"), 1);
    const char *draft_path = fortai_env3("FORTAI_DRAFT_MODEL",
        "LLAMA_ARG_MODEL_DRAFT", "LLAMACPP_DRAFT_MODEL");
    const int requested_spec_max = fortai_parse_int(fortai_env3(
        "FORTAI_SPEC_DRAFT_N_MAX", "LLAMA_ARG_SPEC_DRAFT_N_MAX",
        "LLAMACPP_SPEC_DRAFT_N_MAX"), 2, 1, 32);
    const char *tensor_split = fortai_env3("FORTAI_TENSOR_SPLIT",
        "LLAMA_ARG_TENSOR_SPLIT", "LLAMACPP_TENSOR_SPLIT");
    const float *split_values = fortai_parse_tensor_split(tensor_split);
    if (split_values != NULL && gpu_layers != 0) {
        model_params.split_mode = fortai_parse_int(fortai_env3(
            "FORTAI_SPLIT_MODE", "LLAMA_ARG_SPLIT_MODE", "LLAMACPP_SPLIT_MODE"),
            1, 0, 2);
        model_params.tensor_split = split_values;
    }
    fortai_llama_context_params context_params = handle->api.context_default();
    /* Pass the requested context through unchanged.  llama.cpp applies the
     * same minimum/slot normalization internally as llama-server; doing the
     * padding here would bypass that normalization path. */
    context_params.n_ctx = (uint32_t)context_size;
    /* Keep llama-server's public defaults (n_batch=2048, n_ubatch=512).
     * `-c` changes the KV context but does not rewrite either batch limit;
     * retaining the values returned by llama_context_default_params keeps
     * the scheduler/graph shape identical at higher thread counts. */
    /* FortAI drives one causal sequence at a time.  The server defaults to
     * four slots for concurrent requests, but reserving those slots here
     * needlessly enlarges the recurrent/KV graph and hurts CPU scaling. */
    context_params.n_seq_max = (uint32_t)fortai_parse_int(fortai_env3(
        "FORTAI_PARALLEL", "LLAMA_ARG_PARALLEL", "LLAMACPP_PARALLEL"), 1, 1, 512);
    /* Only the final logit is consumed by FortAI.  Zero means “all outputs”
     * and makes llama.cpp reserve a 256-token output graph for this tiny
     * single-token decode, unlike the server's one-output configuration. */
    context_params.n_outputs_max = draft_path != NULL && draft_path[0] != '\0' &&
        !fortai_path_requests_mtp(draft_path)
        ? (uint32_t)requested_spec_max + 1 : 1;
    context_params.n_outputs_max_per_seq = context_params.n_outputs_max;
    const char *batch_value = fortai_env3("FORTAI_BATCH", "LLAMA_ARG_BATCH", "LLAMACPP_BATCH");
    const char *ubatch_value = fortai_env3("FORTAI_UBATCH", "LLAMA_ARG_UBATCH", "LLAMACPP_UBATCH");
    if (batch_value != NULL)
        context_params.n_batch = (uint32_t)fortai_parse_int(batch_value,
            (int)context_params.n_batch, 1, 1 << 20);
    if (ubatch_value != NULL)
        context_params.n_ubatch = (uint32_t)fortai_parse_int(ubatch_value,
            (int)context_params.n_ubatch, 1, 1 << 20);
    context_params.n_threads = threads;
    context_params.n_threads_batch = fortai_parse_int(fortai_env3(
        "FORTAI_THREADS_BATCH", "LLAMA_ARG_THREADS_BATCH", "LLAMACPP_THREADS_BATCH"),
        threads, 1, 4096);
    /* FortAI measures its own wall-clock loop and does not consume llama's
     * per-node accounting.  Avoid that bookkeeping on the CPU fast path;
     * retain the reference setting for CUDA unless explicitly overridden. */
    context_params.no_perf = gpu_layers == 0 && threads > 1;
    const char *no_perf = getenv("FORTAI_LLAMA_FAST_NO_PERF");
    if (no_perf != NULL) context_params.no_perf = strcmp(no_perf, "0") != 0;
    /* Match llama-server's defaults: KV/KQV offload remains enabled even for
     * CPU-only contexts, while the comparison harness explicitly disables
     * generic host-op offload below.  The CPU backend keeps these tensors on
     * host memory, but the flag still selects the same graph construction and
     * memory path as the reference server. */
    context_params.offload_kqv = fortai_parse_bool(fortai_env3(
        "FORTAI_OFFLOAD_KQV", "LLAMA_ARG_OFFLOAD_KQV", "LLAMACPP_OFFLOAD_KQV"), 1);
    context_params.op_offload = gpu_layers == 0 ? false : true;
    context_params.op_offload = fortai_parse_bool(fortai_env3(
        "FORTAI_OP_OFFLOAD", "LLAMA_ARG_OP_OFFLOAD", "LLAMACPP_OP_OFFLOAD"),
        context_params.op_offload);
    context_params.flash_attn_type = fortai_parse_flash_attn(fortai_env3(
        "FORTAI_FLASH_ATTN", "LLAMA_ARG_FLASH_ATTN", "LLAMACPP_FLASH_ATTN"),
        context_params.flash_attn_type);
    context_params.type_k = fortai_parse_cache_type(fortai_env3(
        "FORTAI_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_K", "LLAMACPP_CACHE_TYPE_K"),
        context_params.type_k);
    context_params.type_v = fortai_parse_cache_type(fortai_env3(
        "FORTAI_CACHE_TYPE_V", "LLAMA_ARG_CACHE_TYPE_V", "LLAMACPP_CACHE_TYPE_V"),
        context_params.type_v);
    /* llama-server leaves the recurrent sliding-window cache compact unless
     * --swa-full is explicitly requested.  The public context default is the
     * opposite, so set this explicitly to keep the graph identical. */
    context_params.swa_full = fortai_parse_bool(fortai_env3(
        "FORTAI_SWA_FULL", "LLAMA_ARG_SWA_FULL", "LLAMACPP_SWA_FULL"), 0);
    context_params.kv_unified = fortai_parse_bool(fortai_env3(
        "FORTAI_KV_UNIFIED", "LLAMA_ARG_KV_UNIFIED", "LLAMACPP_KV_UNIFIED"), 0);
    handle->model = handle->api.model_load(path, model_params);
    if (handle->model != NULL)
        handle->context = handle->api.context_init(handle->model, context_params);
    if (handle->context != NULL)
        handle->batch = handle->api.batch_init(context_size, 0, 1);
    if (handle->model == NULL || handle->context == NULL || handle->batch.token == NULL) {
        fortai_fast_diag(handle->model == NULL ? "target model load failed" :
            (handle->context == NULL ? "target context init failed" : "target batch init failed"));
        if (handle->context != NULL) handle->api.context_free(handle->context);
        if (handle->model != NULL) handle->api.model_free(handle->model);
        free(handle);
        return 3;
    }
    /* llama-server attaches an explicitly configured CPU pool after context
     * construction.  The public API otherwise creates an internal pool with
     * less predictable scheduling, which scales poorly for a single decode
     * stream.  Reproduce the server's default hybrid-polling pool when the
     * symbols are available; older libraries simply keep their auto pool. */
    if (handle->api.attach_threadpool != NULL) {
        fortai_threadpool_params_init_fn params_init =
            (fortai_threadpool_params_init_fn)dlsym(RTLD_DEFAULT,
                "ggml_threadpool_params_init");
        fortai_threadpool_new_fn pool_new =
            (fortai_threadpool_new_fn)dlsym(RTLD_DEFAULT, "ggml_threadpool_new");
        handle->threadpool_free =
            (fortai_threadpool_free_fn)dlsym(RTLD_DEFAULT, "ggml_threadpool_free");
        if (params_init != NULL && pool_new != NULL && handle->threadpool_free != NULL) {
            fortai_threadpool_params params;
            params_init(&params, threads);
            handle->threadpool = pool_new(&params);
            if (handle->threadpool != NULL)
                handle->api.attach_threadpool(handle->context, handle->threadpool, NULL);
        }
    }
    handle->api.set_threads(handle->context, threads, threads);
    const fortai_llama_vocab *vocabulary = handle->api.model_vocab(handle->model);
    handle->vocab = handle->api.vocab_count(vocabulary);
    handle->max_context = context_size;
    handle->path = strdup(path);
    if (handle->vocab <= 0 || handle->path == NULL) {
        handle->api.batch_free(handle->batch);
        handle->api.context_free(handle->context);
        if (handle->threadpool_free != NULL && handle->threadpool != NULL)
            handle->threadpool_free(handle->threadpool);
        handle->api.model_free(handle->model);
        free(handle->path);
        free(handle);
        return 4;
    }
    if (draft_path != NULL && draft_path[0] != '\0') {
        handle->spec_max = requested_spec_max;
        if (handle->api.memory_seq_rm == NULL) {
            fortai_llama_fast_context_destroy(handle);
            return 5;
        }
        if (fortai_path_requests_mtp(draft_path)) {
            /* Qwen3.8's mtp-*.gguf is not a standalone draft transformer:
             * it contains only the appended NextN block.  Current llama.cpp
             * creates a second MTP context against the already-loaded target
             * model and shares its memory/hidden-state stream. */
            if (handle->api.layer_nextn_count == NULL || handle->api.embd_out == NULL ||
                handle->api.set_embeddings_nextn == NULL ||
                handle->api.get_embeddings_nextn == NULL ||
                handle->api.context_init == NULL || handle->api.batch_init == NULL ||
                handle->api.layer_nextn_count(handle->model) <= 0) {
                fortai_fast_diag("MTP API unavailable or target has no NextN layer");
                fortai_llama_fast_context_destroy(handle);
                return 5;
            }
            fortai_llama_context_params mtp_params = context_params;
            mtp_params.ctx_type = 1; /* LLAMA_CONTEXT_TYPE_MTP */
            mtp_params.ctx_other = handle->context;
            mtp_params.n_seq_max = 1;
            mtp_params.n_outputs_max = 1;
            mtp_params.n_outputs_max_per_seq = 1;
            mtp_params.n_rs_seq = 0;
            handle->mtp_context = handle->api.context_init(handle->model, mtp_params);
            handle->mtp_embd = handle->api.embd_out(handle->model);
            if (handle->mtp_context != NULL && handle->mtp_embd > 0)
                handle->mtp_batch = handle->api.batch_init(context_size, handle->mtp_embd, 1);
            /* llama_batch_init allocates either token or embeddings.  MTP
             * needs both, so mirror upstream's small ownership fix. */
            if (handle->mtp_batch.token == NULL && handle->mtp_batch.n_tokens == 0)
                handle->mtp_batch.token = (int32_t *)calloc((size_t)context_size,
                    sizeof(int32_t));
            if (handle->mtp_context == NULL || handle->mtp_batch.embd == NULL ||
                handle->mtp_batch.token == NULL) {
                fortai_fast_diag(handle->mtp_context == NULL ? "MTP context init failed" :
                    (handle->mtp_batch.embd == NULL ? "MTP embedding batch init failed" :
                        "MTP token batch allocation failed"));
                fortai_llama_fast_context_destroy(handle);
                return 6;
            }
            handle->api.set_threads(handle->mtp_context, threads, context_params.n_threads_batch);
            handle->api.set_embeddings_nextn(handle->context, true, false);
            handle->api.set_embeddings_nextn(handle->mtp_context, true, true);
            handle->mtp_enabled = 1;
            handle->spec_enabled = 1;
        } else {
        fortai_llama_model_params draft_model_params = model_params;
        draft_model_params.n_gpu_layers = fortai_parse_int(fortai_env3(
            "FORTAI_DRAFT_GPU_LAYERS", "LLAMA_ARG_DRAFT_GPU_LAYERS",
            "LLAMACPP_DRAFT_GPU_LAYERS"), gpu_layers, -1, 8192);
        fortai_llama_context_params draft_context_params = context_params;
        draft_context_params.n_seq_max = 1;
        draft_context_params.n_outputs_max = (uint32_t)handle->spec_max + 1;
        draft_context_params.n_outputs_max_per_seq = (uint32_t)handle->spec_max + 1;
        handle->draft_model = handle->api.model_load(draft_path, draft_model_params);
        if (handle->draft_model != NULL)
            handle->draft_context = handle->api.context_init(handle->draft_model,
                draft_context_params);
        if (handle->draft_context != NULL)
            handle->draft_batch = handle->api.batch_init(context_size, 0, 1);
        if (handle->draft_model == NULL || handle->draft_context == NULL ||
            handle->draft_batch.token == NULL) {
            fortai_llama_fast_context_destroy(handle);
            return 6;
        }
        handle->api.set_threads(handle->draft_context, threads, context_params.n_threads_batch);
        const fortai_llama_vocab *draft_vocabulary =
            handle->api.model_vocab(handle->draft_model);
        handle->draft_vocab = handle->api.vocab_count(draft_vocabulary);
        if (handle->draft_vocab != handle->vocab) {
            fortai_llama_fast_context_destroy(handle);
            return 7;
        }
        handle->spec_enabled = 1;
        }
    }
    /* Build the single-token graph before the caller's timed decode loop.
     * llama-server performs this work while evaluating the prompt, and its
     * reported generation timing starts with the reusable graph. */
    fortai_warm_context(handle, handle->context, &handle->batch);
    if (handle->spec_enabled)
        if (handle->mtp_enabled == 0)
            fortai_warm_context(handle, handle->draft_context, &handle->draft_batch);
    if (vocab != NULL) *vocab = handle->vocab;
    if (layers != NULL && handle->api.layer_count != NULL)
        *layers = handle->api.layer_count(handle->model);
    *out = handle;
    return 0;
}

int fortai_llama_fast_context_decode(void *opaque, int token, int position,
    float *logits, size_t logits_count) {
    fortai_llama_handle *handle = (fortai_llama_handle *)opaque;
    if (handle == NULL || handle->context == NULL || logits == NULL ||
        token < 0 || position < 0 || position >= handle->max_context ||
        logits_count < (size_t)handle->vocab)
        return 1;
    handle->batch.n_tokens = 1;
    handle->batch.token[0] = token;
    handle->batch.pos[0] = position;
    handle->batch.n_seq_id[0] = 1;
    handle->batch.seq_id[0][0] = 0;
    handle->batch.logits[0] = 1;
    int decode_status = handle->api.decode(handle->context, handle->batch);
    if (decode_status != 0) return 2;
    float *result = handle->api.logits(handle->context, -1);
    if (result == NULL) return 3;
    memcpy(logits, result, (size_t)handle->vocab * sizeof(float));
    return 0;
}

/* Greedy generation does not need a host-side copy of the complete logits
 * vector.  llama.cpp's sampler reads this same resident buffer directly;
 * scan it in place and return the first maximum, matching Fortran MAXLOC's
 * tie rule while avoiding one 1-MiB write and a second full-vector read per
 * token for the Qwen3.5 vocabulary. */
int fortai_llama_fast_context_decode_greedy(void *opaque, int token, int position,
    int *next_token, float *logit_sum) {
    fortai_llama_handle *handle = (fortai_llama_handle *)opaque;
    if (handle == NULL || handle->context == NULL || next_token == NULL ||
        logit_sum == NULL ||
        token < 0 || position < 0 || position >= handle->max_context)
        return 1;
    handle->batch.n_tokens = 1;
    handle->batch.token[0] = token;
    handle->batch.pos[0] = position;
    handle->batch.n_seq_id[0] = 1;
    handle->batch.seq_id[0][0] = 0;
    handle->batch.logits[0] = 1;
    if (handle->api.decode(handle->context, handle->batch) != 0) return 2;
    float *result = handle->api.logits(handle->context, -1);
    if (result == NULL || handle->vocab <= 0) return 3;
    *next_token = fortai_argmax_f32(result, handle->vocab, logit_sum);
    return 0;
}

/* Greedy external-draft speculation.  The target evaluates the current token
 * plus all draft tokens in one batch, then rolls back the unaccepted suffix.
 * The caller consumes the returned prefix as a variable-length generation
 * step; the final returned token is deliberately left unevaluated and becomes
 * the input to the next step, matching llama-server's draft acceptance rule. */
int fortai_llama_fast_context_decode_speculative(void *opaque, int token, int position,
    int *tokens, int capacity, int *count, float *logit_sum) {
    fortai_llama_handle *handle = (fortai_llama_handle *)opaque;
    int draft_tokens[32];
    int accepted = 0;
    int mismatch = 0;
    int i;
    if (handle == NULL || handle->context == NULL || tokens == NULL || count == NULL ||
        logit_sum == NULL || token < 0 || position < 0 || position >= handle->max_context ||
        capacity <= 0)
        return 1;
    if (handle->mtp_enabled && handle->mtp_context != NULL) {
        /* The Qwen3.8 sidecar is a single NextN head, not a standalone draft
         * transformer.  Feed the target hidden row through that head once,
         * and use exact target argmax as the oracle.  The target context is
         * intentionally advanced by one token per call so rollback and
         * recurrent-state semantics remain identical to the greedy path. */
        float *target_hidden;
        float *target_logits;
        float *draft_logits;
        int target_token;
        int draft_token;
        float draft_sum;
        if (capacity < 1 || handle->api.set_embeddings_nextn == NULL ||
            handle->api.get_embeddings_nextn == NULL || handle->mtp_batch.embd == NULL ||
            handle->mtp_batch.token == NULL || handle->mtp_embd <= 0)
            return 1;
        handle->batch.n_tokens = 1;
        handle->batch.token[0] = token;
        handle->batch.pos[0] = position;
        handle->batch.n_seq_id[0] = 1;
        handle->batch.seq_id[0][0] = 0;
        handle->batch.logits[0] = 1;
        if (handle->api.decode(handle->context, handle->batch) != 0) return 2;
        target_logits = handle->api.logits(handle->context, -1);
        target_hidden = handle->api.get_embeddings_nextn(handle->context);
        if (target_logits == NULL || target_hidden == NULL) return 3;
        target_token = fortai_argmax_f32(target_logits, handle->vocab, logit_sum);
        memcpy(handle->mtp_batch.embd, target_hidden,
            (size_t)handle->mtp_embd * sizeof(float));
        handle->mtp_batch.n_tokens = 1;
        handle->mtp_batch.token[0] = token;
        handle->mtp_batch.pos[0] = position;
        handle->mtp_batch.n_seq_id[0] = 1;
        handle->mtp_batch.seq_id[0][0] = 0;
        handle->mtp_batch.logits[0] = 1;
        if (handle->api.decode(handle->mtp_context, handle->mtp_batch) != 0) return 4;
        draft_logits = handle->api.logits(handle->mtp_context, -1);
        if (draft_logits == NULL) return 5;
        draft_token = fortai_argmax_f32(draft_logits, handle->vocab, &draft_sum);
        tokens[0] = draft_token == target_token ? draft_token : target_token;
        *count = 1;
        return 0;
    }
    if (!handle->spec_enabled || handle->spec_max <= 0 || handle->draft_context == NULL ||
        handle->api.memory_seq_rm == NULL || handle->spec_max + 1 > capacity ||
        handle->spec_max + 1 > handle->max_context - position) {
        if (capacity < 1) return 1;
        if (fortai_llama_fast_context_decode_greedy(handle, token, position, &tokens[0],
            logit_sum) != 0)
            return 2;
        *count = 1;
        return 0;
    }

    for (i = 0; i < handle->spec_max; ++i) {
        const int input = i == 0 ? token : draft_tokens[i - 1];
        handle->draft_batch.n_tokens = 1;
        handle->draft_batch.token[0] = input;
        handle->draft_batch.pos[0] = position + i;
        handle->draft_batch.n_seq_id[0] = 1;
        handle->draft_batch.seq_id[0][0] = 0;
        handle->draft_batch.logits[0] = 1;
        if (handle->api.decode(handle->draft_context, handle->draft_batch) != 0)
            return 3;
        float *draft_logits = handle->api.logits(handle->draft_context, -1);
        if (draft_logits == NULL) return 4;
        draft_tokens[i] = fortai_argmax_f32(draft_logits, handle->draft_vocab, logit_sum);
    }

    handle->batch.n_tokens = handle->spec_max + 1;
    for (i = 0; i <= handle->spec_max; ++i) {
        handle->batch.token[i] = i == 0 ? token : draft_tokens[i - 1];
        handle->batch.pos[i] = position + i;
        handle->batch.n_seq_id[i] = 1;
        handle->batch.seq_id[i][0] = 0;
        handle->batch.logits[i] = 1;
    }
    if (handle->api.decode(handle->context, handle->batch) != 0) return 5;
    for (i = 0; i < handle->spec_max; ++i) {
        float *target_logits = handle->api.logits(handle->context, i);
        if (target_logits == NULL) return 6;
        mismatch = fortai_argmax_f32(target_logits, handle->vocab, logit_sum);
        if (mismatch != draft_tokens[i]) break;
        accepted++;
    }
    if (accepted == handle->spec_max) {
        float *target_logits = handle->api.logits(handle->context, handle->spec_max);
        if (target_logits == NULL) return 7;
        mismatch = fortai_argmax_f32(target_logits, handle->vocab, logit_sum);
    }
    if (accepted < handle->spec_max) {
        if (!handle->api.memory_seq_rm(handle->api.memory(handle->context), 0,
            position + accepted + 1, -1)) return 8;
        if (!handle->api.memory_seq_rm(handle->api.memory(handle->draft_context), 0,
            position + accepted + 1, -1)) return 9;
    }
    for (i = 0; i < accepted; ++i) tokens[i] = draft_tokens[i];
    tokens[accepted] = mismatch;
    *count = accepted + 1;
    return 0;
}

int fortai_llama_fast_context_reset(void *opaque) {
    fortai_llama_handle *handle = (fortai_llama_handle *)opaque;
    if (handle == NULL || handle->context == NULL || handle->api.memory == NULL ||
        handle->api.memory_clear == NULL)
        return 1;
    handle->api.memory_clear(handle->api.memory(handle->context), true);
    if (handle->mtp_enabled) {
        if (handle->mtp_context != NULL && handle->api.memory(handle->mtp_context) != NULL)
            handle->api.memory_clear(handle->api.memory(handle->mtp_context), true);
        return 0;
    }
    if (handle->spec_enabled && handle->draft_context != NULL)
        handle->api.memory_clear(handle->api.memory(handle->draft_context), true);
    return 0;
}

int fortai_llama_fast_context_destroy(void *opaque) {
    fortai_llama_handle *handle = (fortai_llama_handle *)opaque;
    if (handle == NULL) return 0;
    if (handle->api.batch_free != NULL && handle->batch.token != NULL)
        handle->api.batch_free(handle->batch);
    if (handle->api.batch_free != NULL && handle->draft_batch.token != NULL)
        handle->api.batch_free(handle->draft_batch);
    if (handle->api.batch_free != NULL && handle->mtp_batch.token != NULL) {
        /* token is added manually because llama_batch_init allocates either
         * token ids or embeddings, never both. */
        free(handle->mtp_batch.token);
        handle->mtp_batch.token = NULL;
        handle->api.batch_free(handle->mtp_batch);
    }
    if (handle->api.context_free != NULL && handle->draft_context != NULL)
        handle->api.context_free(handle->draft_context);
    if (handle->api.context_free != NULL && handle->mtp_context != NULL)
        handle->api.context_free(handle->mtp_context);
    if (handle->api.model_free != NULL && handle->draft_model != NULL)
        handle->api.model_free(handle->draft_model);
    if (handle->api.context_free != NULL && handle->context != NULL)
        handle->api.context_free(handle->context);
    if (handle->threadpool_free != NULL && handle->threadpool != NULL)
        handle->threadpool_free(handle->threadpool);
    if (handle->api.model_free != NULL && handle->model != NULL)
        handle->api.model_free(handle->model);
    free(handle->path);
    free(handle);
    return 0;
}
