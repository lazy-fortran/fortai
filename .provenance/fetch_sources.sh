#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dry_run=false
manifest="$root_dir/.provenance/sources.tsv"
if [[ "${1:-}" == '--dry-run' ]]; then
    dry_run=true
elif [[ -n "${1:-}" ]]; then
    manifest="$1"
fi
source_dir="${FORTAI_PROVENANCE_SOURCE_DIR:-$root_dir/.provenance/src}"
record_dir="${FORTAI_PROVENANCE_RECORD_DIR:-$root_dir/.provenance/records}"

if [[ "$dry_run" == false ]]; then
    mkdir -p "$source_dir" "$record_dir"
fi
while IFS=$'\t' read -r name url ref license; do
    if [[ -z "$name" || "$name" == \#* ]]; then
        continue
    fi
    if [[ "$dry_run" == true ]]; then
        printf '%s\t%s\t%s\t%s\n' "$name" "$url" "$ref" "$license"
        continue
    fi
    target="$source_dir/$name"
    if [[ -d "$target/.git" ]]; then
        git -C "$target" fetch --depth=1 origin "$ref"
        git -C "$target" checkout --detach FETCH_HEAD
    else
        git clone --filter=blob:none --depth=1 --branch "$ref" "$url" "$target"
    fi
    commit=$(git -C "$target" rev-parse HEAD)
    printf 'name=%s\nurl=%s\nref=%s\nlicense=%s\ncommit=%s\n' \
        "$name" "$url" "$ref" "$license" "$commit" >"$record_dir/$name.txt"
    printf '%s %s %s\n' "$name" "$commit" "$license"
done <"$manifest"
