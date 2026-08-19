#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
device=${CUDA_DEVICE:-0}
rows=${FORTAI_CUDA_ROWS:-4096}
width=${FORTAI_CUDA_WIDTH:-4096}
iterations=${FORTAI_CUDA_ITERATIONS:-1000}
warmup=${FORTAI_CUDA_WARMUP:-100}
seed=${FORTAI_CUDA_SEED:-0x6f727461}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
profile_dir="$root_dir/benchmark/profiles/cuda_q8_${rows}x${width}_d${device}_${stamp}"
mkdir -p "$profile_dir"
args=(--device "$device" --rows "$rows" --width "$width" --iterations "$iterations" --warmup "$warmup" --seed "$seed")

"$root_dir/tools/build_cuda.sh" >"$profile_dir/build-fortai.log"
"$root_dir/tools/build_llama_cuda_q8.sh" >"$profile_dir/build-llama.log"
"$root_dir/tools/worktree_digest.sh" >"$profile_dir/worktree.txt"
{
    printf 'commit=%s\nmodel_scope=resident_q8_gemv_microbenchmark\n' "$(git -C "$root_dir" rev-parse HEAD)"
    printf 'device=%s\nrows=%s\nwidth=%s\niterations=%s\nwarmup=%s\nseed=%s\n' \
        "$device" "$rows" "$width" "$iterations" "$warmup" "$seed"
    printf 'nvcc=%s\n' "$(/opt/cuda/bin/nvcc --version | tail -n 1)"
    printf 'ncu=%s\n' "$(ncu --version | head -n 1)"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader
} >"$profile_dir/provenance.txt"

"$root_dir/build/cuda/fortai_cuda_q8_bench" "${args[@]}" >"$profile_dir/fortai-run.json"
llama_lib=${LLAMA_LIBRARY_DIR:-/home/ert/.local/llama.cpp}
LD_LIBRARY_PATH="$llama_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$root_dir/build/cuda/llama_cuda_q8_bench" "${args[@]}" >"$profile_dir/llama-run.json"

# Raw Nsight Compute CSV is retained so the exact metric selection and all
# replay/counter diagnostics remain inspectable after this shell exits.
set +e
ncu --target-processes all --csv --page raw --set full \
    --launch-skip "$warmup" --launch-count 1 \
    "$root_dir/build/cuda/fortai_cuda_q8_bench" "${args[@]}" \
    >"$profile_dir/fortai-ncu.csv" 2>&1
fortai_ncu_status=$?
LD_LIBRARY_PATH="$llama_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ncu --target-processes all --csv --page raw --set full \
    --launch-skip "$warmup" --launch-count 1 \
    "$root_dir/build/cuda/llama_cuda_q8_bench" "${args[@]}" \
    >"$profile_dir/llama-ncu.csv" 2>&1
llama_ncu_status=$?
set -e
printf 'fortai_ncu_status=%s\nllama_ncu_status=%s\n' "$fortai_ncu_status" "$llama_ncu_status" \
    >"$profile_dir/profiling-status.txt"

nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
    -o "$profile_dir/fortai-nsys" \
    "$root_dir/build/cuda/fortai_cuda_q8_bench" "${args[@]}" \
    >"$profile_dir/fortai-nsys.log" 2>&1
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format csv --force-export=true \
    "$profile_dir/fortai-nsys.nsys-rep" >"$profile_dir/fortai-nsys-stats.csv" 2>&1 || true
LD_LIBRARY_PATH="$llama_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
    -o "$profile_dir/llama-nsys" \
    "$root_dir/build/cuda/llama_cuda_q8_bench" "${args[@]}" \
    >"$profile_dir/llama-nsys.log" 2>&1
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format csv --force-export=true \
    "$profile_dir/llama-nsys.nsys-rep" >"$profile_dir/llama-nsys-stats.csv" 2>&1 || true

printf 'profile_dir=%s\nfortai_ncu=%s\nllama_ncu=%s\n' \
    "$profile_dir" "$profile_dir/fortai-ncu.csv" "$profile_dir/llama-ncu.csv"
