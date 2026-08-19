#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct options_t {
    int device = 0;
    int heads = 16;
    int head_size = 128;
    int tokens = 128;
    int warmup = 32;
    int repeats = 9;
    uint32_t seed = 0x35a08bU;
    float absolute_tolerance = 2.0e-5f;
    float relative_tolerance = 3.0e-4f;
};

struct comparison_t {
    float maximum_absolute = 0.0f;
    float maximum_relative = 0.0f;
    size_t violations = 0;
};

struct timing_t {
    std::vector<float> samples_ms;
    float minimum_ms = 0.0f;
    float median_ms = 0.0f;
    float maximum_ms = 0.0f;
};

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(EXIT_FAILURE);
}

void check_cuda(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

int parse_int(const char *text, const char *name) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || !end || *end || value <= 0 || value > 1'000'000)
        fail((std::string("invalid ") + name + ": " + text).c_str());
    return static_cast<int>(value);
}

int parse_device(const char *text) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || !end || *end || value < 0 || value > 1024)
        fail((std::string("invalid device: ") + text).c_str());
    return static_cast<int>(value);
}

options_t parse_options(int argc, char **argv) {
    options_t options;
    for (int i = 1; i < argc; ++i) {
        auto value = [&](const char *name) {
            if (++i >= argc) fail((std::string("missing value for ") + name).c_str());
            return argv[i];
        };
        if (!std::strcmp(argv[i], "--device"))
            options.device = parse_device(value("--device"));
        else if (!std::strcmp(argv[i], "--heads"))
            options.heads = parse_int(value("--heads"), "heads");
        else if (!std::strcmp(argv[i], "--head-size"))
            options.head_size = parse_int(value("--head-size"), "head size");
        else if (!std::strcmp(argv[i], "--tokens"))
            options.tokens = parse_int(value("--tokens"), "tokens");
        else if (!std::strcmp(argv[i], "--warmup"))
            options.warmup = parse_int(value("--warmup"), "warmup");
        else if (!std::strcmp(argv[i], "--repeats"))
            options.repeats = parse_int(value("--repeats"), "repeats");
        else if (!std::strcmp(argv[i], "--seed"))
            options.seed = static_cast<uint32_t>(parse_int(value("--seed"), "seed"));
        else if (!std::strcmp(argv[i], "--help")) {
            std::puts("fortai_cuda_gdn_ar_bench [--device 0] [--heads 16] "
                "[--head-size 128] [--tokens 128] [--warmup 32] [--repeats 9]");
            std::exit(EXIT_SUCCESS);
        } else {
            fail((std::string("unknown option: ") + argv[i]).c_str());
        }
    }
    if (options.head_size > 1024 || (options.head_size & (options.head_size - 1)) != 0)
        fail("head size must be a power of two no larger than 1024");
    return options;
}

std::string json_escape(const char *text) {
    if (!text) return "";
    std::string escaped;
    for (const unsigned char value : std::string(text)) {
        switch (value) {
        case '\\': escaped += "\\\\"; break;
        case '"': escaped += "\\\""; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (value >= 0x20) escaped += static_cast<char>(value);
        }
    }
    return escaped;
}

const char *environment(const char *name) {
    const char *value = std::getenv(name);
    return value ? value : "";
}

size_t state_index(int head, int row, int column, int head_size) {
    return (static_cast<size_t>(head) * head_size + row) * head_size + column;
}

size_t vector_index(int token, int head, int element, int heads, int head_size) {
    return (static_cast<size_t>(token) * heads + head) * head_size + element;
}

void normalize_heads(std::vector<float> &values, int tokens, int heads, int head_size) {
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            double norm = 0.0;
            for (int i = 0; i < head_size; ++i) {
                const float value = values[vector_index(token, head, i, heads, head_size)];
                norm += static_cast<double>(value) * value;
            }
            const float inverse = 1.0f / std::sqrt(static_cast<float>(norm) + 1.0e-6f);
            for (int i = 0; i < head_size; ++i)
                values[vector_index(token, head, i, heads, head_size)] *= inverse;
        }
    }
}

void make_inputs(const options_t &options, std::vector<float> &q, std::vector<float> &k,
    std::vector<float> &v, std::vector<float> &g, std::vector<float> &beta) {
    std::mt19937 generator(options.seed);
    std::uniform_real_distribution<float> vector_distribution(-1.0f, 1.0f);
    std::uniform_real_distribution<float> value_distribution(-0.5f, 0.5f);
    std::uniform_real_distribution<float> gate_distribution(-0.04f, -0.005f);
    std::uniform_real_distribution<float> beta_distribution(0.2f, 0.8f);
    for (float &value : q) value = vector_distribution(generator);
    for (float &value : k) value = vector_distribution(generator);
    for (float &value : v) value = value_distribution(generator);
    for (float &value : g) value = gate_distribution(generator);
    for (float &value : beta) value = beta_distribution(generator);
    normalize_heads(q, options.tokens, options.heads, options.head_size);
    normalize_heads(k, options.tokens, options.heads, options.head_size);
}

// Independent scalar CPU recurrence. State is stored transposed by output row,
// matching the mathematical update S[:,j] and not either CUDA launch schedule.
void cpu_oracle(const options_t &options, const std::vector<float> &q,
    const std::vector<float> &k, const std::vector<float> &v,
    const std::vector<float> &g, const std::vector<float> &beta,
    std::vector<float> &state, std::vector<float> &output) {
    const float output_scale = 1.0f / std::sqrt(static_cast<float>(options.head_size));
    for (int token = 0; token < options.tokens; ++token) {
        for (int head = 0; head < options.heads; ++head) {
            const float decay = std::exp(g[static_cast<size_t>(token) * options.heads + head]);
            for (int row = 0; row < options.head_size; ++row) {
                float prediction = 0.0f;
                for (int column = 0; column < options.head_size; ++column) {
                    float &entry = state[state_index(head, row, column, options.head_size)];
                    entry *= decay;
                    prediction += entry * k[vector_index(token, head, column,
                        options.heads, options.head_size)];
                }
                const float delta = (v[vector_index(token, head, row, options.heads,
                    options.head_size)] - prediction) *
                    beta[static_cast<size_t>(token) * options.heads + head];
                float projected = 0.0f;
                for (int column = 0; column < options.head_size; ++column) {
                    float &entry = state[state_index(head, row, column, options.head_size)];
                    entry += delta * k[vector_index(token, head, column,
                        options.heads, options.head_size)];
                    projected += entry * q[vector_index(token, head, column,
                        options.heads, options.head_size)];
                }
                output[vector_index(token, head, row, options.heads,
                    options.head_size)] = projected * output_scale;
            }
        }
    }
}

__global__ void scale_state_kernel(float *__restrict__ state,
    const float *__restrict__ g, int heads, int head_size) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(heads) * head_size * head_size;
    if (index >= count) return;
    const int head = static_cast<int>(index / (head_size * head_size));
    state[index] *= expf(g[head]);
}

__global__ void delta_kernel(const float *__restrict__ state,
    const float *__restrict__ k, const float *__restrict__ v,
    const float *__restrict__ beta, float *__restrict__ delta, int head_size) {
    extern __shared__ float reduction[];
    const int row_index = static_cast<int>(blockIdx.x);
    const int head = row_index / head_size;
    const int row = row_index % head_size;
    const int column = static_cast<int>(threadIdx.x);
    const size_t state_offset = static_cast<size_t>(row_index) * head_size;
    reduction[column] = state[state_offset + column] *
        k[static_cast<size_t>(head) * head_size + column];
    __syncthreads();
    for (int stride = head_size / 2; stride > 0; stride >>= 1) {
        if (column < stride) reduction[column] += reduction[column + stride];
        __syncthreads();
    }
    if (column == 0)
        delta[row_index] = (v[static_cast<size_t>(head) * head_size + row] - reduction[0]) *
            beta[head];
}

__global__ void update_state_kernel(float *__restrict__ state,
    const float *__restrict__ k, const float *__restrict__ delta,
    int heads, int head_size) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(heads) * head_size * head_size;
    if (index >= count) return;
    const int row_index = static_cast<int>(index / head_size);
    const int head = row_index / head_size;
    const int column = static_cast<int>(index % head_size);
    state[index] += delta[row_index] * k[static_cast<size_t>(head) * head_size + column];
}

__global__ void output_kernel(const float *__restrict__ state,
    const float *__restrict__ q, float *__restrict__ output, int head_size,
    float output_scale) {
    extern __shared__ float reduction[];
    const int row_index = static_cast<int>(blockIdx.x);
    const int head = row_index / head_size;
    const int column = static_cast<int>(threadIdx.x);
    const size_t state_offset = static_cast<size_t>(row_index) * head_size;
    reduction[column] = state[state_offset + column] *
        q[static_cast<size_t>(head) * head_size + column];
    __syncthreads();
    for (int stride = head_size / 2; stride > 0; stride >>= 1) {
        if (column < stride) reduction[column] += reduction[column + stride];
        __syncthreads();
    }
    if (column == 0) output[row_index] = reduction[0] * output_scale;
}

__global__ void fused_gdn_ar_kernel(float *__restrict__ state,
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, float *__restrict__ output,
    int head_size, float output_scale) {
    extern __shared__ float reduction[];
    __shared__ float row_delta;
    const int row_index = static_cast<int>(blockIdx.x);
    const int head = row_index / head_size;
    const int row = row_index % head_size;
    const int column = static_cast<int>(threadIdx.x);
    const size_t state_offset = static_cast<size_t>(row_index) * head_size;
    const size_t vector_offset = static_cast<size_t>(head) * head_size;
    float entry = state[state_offset + column] * expf(g[head]);
    state[state_offset + column] = entry;
    reduction[column] = entry * k[vector_offset + column];
    __syncthreads();
    for (int stride = head_size / 2; stride > 0; stride >>= 1) {
        if (column < stride) reduction[column] += reduction[column + stride];
        __syncthreads();
    }
    if (column == 0)
        row_delta = (v[vector_offset + row] - reduction[0]) * beta[head];
    __syncthreads();
    entry += row_delta * k[vector_offset + column];
    state[state_offset + column] = entry;
    reduction[column] = entry * q[vector_offset + column];
    __syncthreads();
    for (int stride = head_size / 2; stride > 0; stride >>= 1) {
        if (column < stride) reduction[column] += reduction[column + stride];
        __syncthreads();
    }
    if (column == 0) output[row_index] = reduction[0] * output_scale;
}

void launch_decomposed(const options_t &options, int token, float *state,
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, float *delta, float *output, cudaStream_t stream) {
    const int inner = options.heads * options.head_size;
    const size_t state_count = static_cast<size_t>(inner) * options.head_size;
    const int state_blocks = static_cast<int>((state_count + 255) / 256);
    const size_t vector_offset = static_cast<size_t>(token) * inner;
    const size_t scalar_offset = static_cast<size_t>(token) * options.heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(options.head_size));
    scale_state_kernel<<<state_blocks, 256, 0, stream>>>(state, g + scalar_offset,
        options.heads, options.head_size);
    delta_kernel<<<inner, options.head_size, options.head_size * sizeof(float), stream>>>(
        state, k + vector_offset, v + vector_offset, beta + scalar_offset,
        delta, options.head_size);
    update_state_kernel<<<state_blocks, 256, 0, stream>>>(state, k + vector_offset,
        delta, options.heads, options.head_size);
    output_kernel<<<inner, options.head_size, options.head_size * sizeof(float), stream>>>(
        state, q + vector_offset, output + vector_offset, options.head_size, scale);
}

void launch_fused(const options_t &options, int token, float *state,
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, float *output, cudaStream_t stream) {
    const int inner = options.heads * options.head_size;
    const size_t vector_offset = static_cast<size_t>(token) * inner;
    const size_t scalar_offset = static_cast<size_t>(token) * options.heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(options.head_size));
    fused_gdn_ar_kernel<<<inner, options.head_size,
        options.head_size * sizeof(float), stream>>>(state, q + vector_offset,
        k + vector_offset, v + vector_offset, g + scalar_offset,
        beta + scalar_offset, output + vector_offset, options.head_size, scale);
}

comparison_t compare(const std::vector<float> &actual, const std::vector<float> &expected,
    float absolute_tolerance, float relative_tolerance) {
    comparison_t error;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float absolute = std::abs(actual[i] - expected[i]);
        const float relative = absolute / std::max(std::abs(expected[i]), 1.0e-6f);
        error.maximum_absolute = std::max(error.maximum_absolute, absolute);
        error.maximum_relative = std::max(error.maximum_relative, relative);
        if (absolute > absolute_tolerance + relative_tolerance * std::abs(expected[i]))
            ++error.violations;
    }
    return error;
}

timing_t summarize(std::vector<float> samples) {
    timing_t result;
    result.samples_ms = samples;
    std::sort(samples.begin(), samples.end());
    result.minimum_ms = samples.front();
    result.median_ms = samples[samples.size() / 2];
    result.maximum_ms = samples.back();
    return result;
}

} // namespace

int main(int argc, char **argv) {
    const options_t options = parse_options(argc, argv);
    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, options.device), "cudaGetDeviceProperties");
    int driver_version = 0;
    int runtime_version = 0;
    check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");

    const int inner = options.heads * options.head_size;
    const size_t state_count = static_cast<size_t>(inner) * options.head_size;
    const size_t vector_count = static_cast<size_t>(options.tokens) * inner;
    const size_t scalar_count = static_cast<size_t>(options.tokens) * options.heads;
    std::vector<float> q(vector_count), k(vector_count), v(vector_count);
    std::vector<float> g(scalar_count), beta(scalar_count);
    std::vector<float> initial_state(state_count, 0.0f);
    std::vector<float> oracle_state = initial_state;
    std::vector<float> oracle_output(vector_count);
    make_inputs(options, q, k, v, g, beta);
    const auto cpu_start = std::chrono::steady_clock::now();
    cpu_oracle(options, q, k, v, g, beta, oracle_state, oracle_output);
    const double cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cpu_start).count();

    float *device_q = nullptr;
    float *device_k = nullptr;
    float *device_v = nullptr;
    float *device_g = nullptr;
    float *device_beta = nullptr;
    float *device_initial_state = nullptr;
    float *device_state = nullptr;
    float *device_delta = nullptr;
    float *device_output = nullptr;
    check_cuda(cudaMalloc(&device_q, vector_count * sizeof(float)), "cudaMalloc q");
    check_cuda(cudaMalloc(&device_k, vector_count * sizeof(float)), "cudaMalloc k");
    check_cuda(cudaMalloc(&device_v, vector_count * sizeof(float)), "cudaMalloc v");
    check_cuda(cudaMalloc(&device_g, scalar_count * sizeof(float)), "cudaMalloc g");
    check_cuda(cudaMalloc(&device_beta, scalar_count * sizeof(float)), "cudaMalloc beta");
    check_cuda(cudaMalloc(&device_initial_state, state_count * sizeof(float)),
        "cudaMalloc initial state");
    check_cuda(cudaMalloc(&device_state, state_count * sizeof(float)), "cudaMalloc state");
    check_cuda(cudaMalloc(&device_delta, inner * sizeof(float)), "cudaMalloc delta");
    check_cuda(cudaMalloc(&device_output, vector_count * sizeof(float)), "cudaMalloc output");
    check_cuda(cudaMemcpy(device_q, q.data(), vector_count * sizeof(float),
        cudaMemcpyHostToDevice), "copy q");
    check_cuda(cudaMemcpy(device_k, k.data(), vector_count * sizeof(float),
        cudaMemcpyHostToDevice), "copy k");
    check_cuda(cudaMemcpy(device_v, v.data(), vector_count * sizeof(float),
        cudaMemcpyHostToDevice), "copy v");
    check_cuda(cudaMemcpy(device_g, g.data(), scalar_count * sizeof(float),
        cudaMemcpyHostToDevice), "copy g");
    check_cuda(cudaMemcpy(device_beta, beta.data(), scalar_count * sizeof(float),
        cudaMemcpyHostToDevice), "copy beta");
    check_cuda(cudaMemcpy(device_initial_state, initial_state.data(), state_count * sizeof(float),
        cudaMemcpyHostToDevice), "copy initial state");

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    auto reset_state = [&]() {
        check_cuda(cudaMemcpyAsync(device_state, device_initial_state,
            state_count * sizeof(float), cudaMemcpyDeviceToDevice, stream), "reset state");
        check_cuda(cudaMemsetAsync(device_output, 0, vector_count * sizeof(float), stream),
            "reset output");
    };
    auto launch_sequence = [&](bool fused, int tokens) {
        for (int token = 0; token < tokens; ++token) {
            if (fused)
                launch_fused(options, token, device_state, device_q, device_k, device_v,
                    device_g, device_beta, device_output, stream);
            else
                launch_decomposed(options, token, device_state, device_q, device_k, device_v,
                    device_g, device_beta, device_delta, device_output, stream);
        }
        check_cuda(cudaGetLastError(), fused ? "launch fused" : "launch decomposed");
    };
    auto collect = [&](bool fused, std::vector<float> &state, std::vector<float> &output) {
        reset_state();
        launch_sequence(fused, options.tokens);
        check_cuda(cudaMemcpyAsync(state.data(), device_state, state_count * sizeof(float),
            cudaMemcpyDeviceToHost, stream), "collect state");
        check_cuda(cudaMemcpyAsync(output.data(), device_output, vector_count * sizeof(float),
            cudaMemcpyDeviceToHost, stream), "collect output");
        check_cuda(cudaStreamSynchronize(stream), "collect synchronize");
    };

    std::vector<float> decomposed_state(state_count), fused_state(state_count);
    std::vector<float> decomposed_output(vector_count), fused_output(vector_count);
    collect(false, decomposed_state, decomposed_output);
    collect(true, fused_state, fused_output);
    const comparison_t decomposed_output_error = compare(decomposed_output, oracle_output,
        options.absolute_tolerance, options.relative_tolerance);
    const comparison_t decomposed_state_error = compare(decomposed_state, oracle_state,
        options.absolute_tolerance, options.relative_tolerance);
    const comparison_t fused_output_error = compare(fused_output, oracle_output,
        options.absolute_tolerance, options.relative_tolerance);
    const comparison_t fused_state_error = compare(fused_state, oracle_state,
        options.absolute_tolerance, options.relative_tolerance);
    const comparison_t cross_output_error = compare(fused_output, decomposed_output,
        options.absolute_tolerance, options.relative_tolerance);

    reset_state();
    for (int token = 0; token < options.warmup; ++token) {
        const int input_token = token % options.tokens;
        launch_fused(options, input_token, device_state, device_q, device_k, device_v,
            device_g, device_beta, device_output, stream);
        launch_decomposed(options, input_token, device_state, device_q, device_k, device_v,
            device_g, device_beta, device_delta, device_output, stream);
    }
    check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

    std::vector<float> decomposed_samples;
    std::vector<float> fused_samples;
    auto time_sequence = [&](bool fused) {
        reset_state();
        check_cuda(cudaEventRecord(start, stream), "record start");
        launch_sequence(fused, options.tokens);
        check_cuda(cudaEventRecord(stop, stream), "record stop");
        check_cuda(cudaEventSynchronize(stop), "timing synchronize");
        float elapsed = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "event elapsed time");
        return elapsed;
    };
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        if ((repeat & 1) == 0) {
            decomposed_samples.push_back(time_sequence(false));
            fused_samples.push_back(time_sequence(true));
        } else {
            fused_samples.push_back(time_sequence(true));
            decomposed_samples.push_back(time_sequence(false));
        }
    }
    const timing_t decomposed_timing = summarize(decomposed_samples);
    const timing_t fused_timing = summarize(fused_samples);
    const float decomposed_us_per_token = 1000.0f * decomposed_timing.median_ms / options.tokens;
    const float fused_us_per_token = 1000.0f * fused_timing.median_ms / options.tokens;
    const bool correct = decomposed_output_error.violations == 0 &&
        decomposed_state_error.violations == 0 && fused_output_error.violations == 0 &&
        fused_state_error.violations == 0 && cross_output_error.violations == 0;

    std::printf("provenance={\"benchmark\":\"fortai_cuda_gdn_ar\","
        "\"git_commit\":\"%s\",\"tracked_patch_sha256\":\"%s\","
        "\"source_sha256\":\"%s\",\"runner_sha256\":\"%s\","
        "\"fixture\":\"%s\","
        "\"fixture_sha256\":\"%s\",\"fixture_dimensions_discovered\":\"%s\","
        "\"build\":\"%s\",\"nvcc\":\"%s\",\"utc\":\"%s\","
        "\"gpu\":\"%s\",\"compute_capability\":\"%d.%d\","
        "\"sm_count\":%d,\"global_memory_bytes\":%zu,"
        "\"cuda_driver\":%d,\"cuda_runtime\":%d}\n",
        json_escape(environment("FORTAI_BENCH_GIT_COMMIT")).c_str(),
        json_escape(environment("FORTAI_BENCH_PATCH_SHA")).c_str(),
        json_escape(environment("FORTAI_BENCH_SOURCE_SHA")).c_str(),
        json_escape(environment("FORTAI_BENCH_RUNNER_SHA")).c_str(),
        json_escape(environment("FORTAI_BENCH_FIXTURE")).c_str(),
        json_escape(environment("FORTAI_BENCH_FIXTURE_SHA")).c_str(),
        json_escape(environment("FORTAI_BENCH_FIXTURE_DISCOVERED")).c_str(),
        json_escape(environment("FORTAI_BENCH_BUILD")).c_str(),
        json_escape(environment("FORTAI_BENCH_NVCC")).c_str(),
        json_escape(environment("FORTAI_BENCH_UTC")).c_str(),
        json_escape(properties.name).c_str(), properties.major, properties.minor,
        properties.multiProcessorCount, properties.totalGlobalMem, driver_version, runtime_version);
    std::printf("dimensions={\"heads\":%d,\"head_size\":%d,\"inner_size\":%d,"
        "\"state_elements\":%zu,\"state_bytes\":%zu,\"tokens\":%d,"
        "\"warmup_tokens\":%d,\"repeats\":%d,\"seed\":%u}\n",
        options.heads, options.head_size, inner, state_count, state_count * sizeof(float),
        options.tokens, options.warmup, options.repeats, options.seed);
    std::printf("correctness={\"status\":\"%s\",\"absolute_tolerance\":%.9g,"
        "\"relative_tolerance\":%.9g,\"cpu_oracle_ms\":%.6f,"
        "\"decomposed_output_max_abs\":%.9g,\"decomposed_output_max_rel\":%.9g,"
        "\"decomposed_output_violations\":%zu,\"decomposed_state_max_abs\":%.9g,"
        "\"decomposed_state_violations\":%zu,\"fused_output_max_abs\":%.9g,"
        "\"fused_output_max_rel\":%.9g,\"fused_output_violations\":%zu,"
        "\"fused_state_max_abs\":%.9g,\"fused_state_violations\":%zu,"
        "\"fused_vs_decomposed_max_abs\":%.9g,\"fused_vs_decomposed_violations\":%zu}\n",
        correct ? "PASS" : "FAIL", options.absolute_tolerance,
        options.relative_tolerance, cpu_ms, decomposed_output_error.maximum_absolute,
        decomposed_output_error.maximum_relative, decomposed_output_error.violations,
        decomposed_state_error.maximum_absolute, decomposed_state_error.violations,
        fused_output_error.maximum_absolute, fused_output_error.maximum_relative,
        fused_output_error.violations, fused_state_error.maximum_absolute,
        fused_state_error.violations, cross_output_error.maximum_absolute,
        cross_output_error.violations);
    std::printf("timing={\"method\":\"cuda_events_device_only\","
        "\"decomposed_launches_per_token\":4,\"fused_launches_per_token\":1,"
        "\"decomposed_median_ms\":%.6f,\"decomposed_min_ms\":%.6f,"
        "\"decomposed_max_ms\":%.6f,\"decomposed_us_per_token\":%.6f,"
        "\"fused_median_ms\":%.6f,\"fused_min_ms\":%.6f,"
        "\"fused_max_ms\":%.6f,\"fused_us_per_token\":%.6f,"
        "\"fused_speedup\":%.6f}\n",
        decomposed_timing.median_ms, decomposed_timing.minimum_ms,
        decomposed_timing.maximum_ms, decomposed_us_per_token,
        fused_timing.median_ms, fused_timing.minimum_ms, fused_timing.maximum_ms,
        fused_us_per_token, decomposed_timing.median_ms / fused_timing.median_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(device_q);
    cudaFree(device_k);
    cudaFree(device_v);
    cudaFree(device_g);
    cudaFree(device_beta);
    cudaFree(device_initial_state);
    cudaFree(device_state);
    cudaFree(device_delta);
    cudaFree(device_output);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
