#!/usr/bin/env bash
set -euo pipefail

# Read-only proof that the sole resident llama-server is the protected GPU
# service.  This helper never connects to its port and never changes its
# process state; callers use the returned PID only for provenance.
protected_pid="${FORTAI_PROTECTED_LLAMA_PID:-268006}"
protected_executable="${FORTAI_PROTECTED_LLAMA_EXECUTABLE:-/home/ert/.local/llama.cpp-b10566-cuda/llama-server}"
protected_port="${FORTAI_PROTECTED_LLAMA_PORT:-8080}"
protected_model="${FORTAI_PROTECTED_LLAMA_MODEL:-Qwen3.8-27B-UD-Q4_K_XL.gguf}"

python3 - "$protected_pid" "$protected_executable" "$protected_port" "$protected_model" <<'PY'
import os
import sys
from pathlib import Path

pid_text, expected_executable, expected_port, expected_model = sys.argv[1:]
try:
    expected_pid = int(pid_text)
except ValueError:
    raise SystemExit("protected llama PID must be an integer")

proc_root = Path("/proc")
pids = []
for entry in proc_root.iterdir():
    if not entry.name.isdigit():
        continue
    try:
        if (entry / "comm").read_text().strip() == "llama-server":
            pids.append(int(entry.name))
    except (FileNotFoundError, PermissionError):
        continue

if pids != [expected_pid]:
    raise SystemExit(
        "resident llama-server set is not exactly the protected PID: "
        f"expected={[expected_pid]} observed={sorted(pids)}"
    )

proc = proc_root / str(expected_pid)
try:
    executable = str((proc / "exe").resolve(strict=True))
    arguments = (proc / "cmdline").read_bytes().split(b"\0")
    arguments = [item.decode(errors="replace") for item in arguments if item]
except (FileNotFoundError, PermissionError) as error:
    raise SystemExit(f"cannot inspect protected llama-server: {error}")

if executable != expected_executable:
    raise SystemExit(
        f"protected llama executable mismatch: expected={expected_executable} "
        f"observed={executable}"
    )
if not arguments or arguments[0] != expected_executable:
    raise SystemExit("protected llama command line does not start with its executable")

def option_value(*names):
    for index, argument in enumerate(arguments[:-1]):
        if argument in names:
            return arguments[index + 1]
    return None

if option_value("--port") != expected_port:
    raise SystemExit("protected llama server is not bound to the protected port")
if option_value("--host") != "0.0.0.0":
    raise SystemExit("protected llama server is not the expected resident host")
if option_value("-ngl", "--n-gpu-layers") != "99":
    raise SystemExit("protected llama server is not configured for GPU residency")
model_path = option_value("-m", "--model")
if model_path is None or Path(model_path).name != expected_model:
    raise SystemExit(
        "protected llama model mismatch: "
        f"expected basename={expected_model} observed={model_path}"
    )

print(expected_pid)
PY
