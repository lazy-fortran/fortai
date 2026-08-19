#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$root_dir/.provenance/models.tsv"
model_dir="${FORTAI_MODEL_DIR:-$root_dir/.provenance/downloads}"
hf_cli="${HF_CLI:-hf}"
dry_run=false
if [[ "${1:-}" == '--dry-run' ]]; then
    dry_run=true
elif [[ -n "${1:-}" ]]; then
    manifest="$1"
fi

if [[ "$dry_run" == false ]] && ! command -v "$hf_cli" >/dev/null 2>&1; then
    echo "Hugging Face CLI not found: $hf_cli" >&2
    exit 2
fi
if [[ "$dry_run" == false ]]; then
    mkdir -p "$model_dir"
fi

while IFS=$'\t' read -r name repository filename revision license; do
    if [[ -z "$name" || "$name" == \#* ]]; then
        continue
    fi
    target="$model_dir/$name"
    if [[ "$dry_run" == true ]]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$repository" "$filename" "$revision" "$license"
        continue
    fi
    mkdir -p "$target"
    "$hf_cli" download "$repository" "$filename" --revision "$revision" --local-dir "$target"
    sha256sum "$target/$filename"
done <"$manifest"
