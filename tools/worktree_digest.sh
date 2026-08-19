#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Refresh Git's stat cache before asking for a patch. The digests below are
# content-based and cover both staged and unstaged changes, even if filesystem
# mtimes are stale.
git -C "$root_dir" update-index --refresh -q >/dev/null 2>&1 || true
patch_digest=$(python3 - "$root_dir" <<'PY'
import hashlib
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()
digest.update(subprocess.check_output(
    ['git', '-C', str(root), 'diff', 'HEAD', '--binary', '--no-ext-diff', '--no-renames'],
    text=False,
))
status = subprocess.check_output(
    ['git', '-C', str(root), 'status', '--porcelain=v1', '-z', '--untracked-files=all'],
    text=False,
)
for entry in status.split(b'\0'):
    if entry.startswith(b'?? '):
        relative = entry[3:]
        path = root / relative.decode()
        digest.update(b'untracked\0' + relative + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)
tree_digest=$(python3 - "$root_dir" <<'PY'
import hashlib
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = subprocess.check_output(
    ['git', '-C', str(root), 'ls-files', '-z'], text=False).split(b'\0')
digest = hashlib.sha256()
for encoded in paths:
    if not encoded:
        continue
    path = root / os.fsdecode(encoded)
    digest.update(encoded)
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)
worktree_digest=$(python3 - "$root_dir" <<'PY'
import hashlib
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = subprocess.check_output(
    ['git', '-C', str(root), 'ls-files', '-z'], text=False).split(b'\0')
status = subprocess.check_output(
    ['git', '-C', str(root), 'status', '--porcelain=v1', '-z', '--untracked-files=all'],
    text=False,
)
for entry in status.split(b'\0'):
    if entry.startswith(b'?? '):
        paths.append(entry[3:])
digest = hashlib.sha256()
for encoded in sorted(set(path for path in paths if path)):
    digest.update(encoded)
    digest.update(hashlib.sha256((root / encoded.decode()).read_bytes()).digest())
print(digest.hexdigest())
PY
)
printf 'patch_digest=%s\n' "$patch_digest"
printf 'tracked_tree_digest=%s\n' "$tree_digest"
printf 'worktree_digest=%s\n' "$worktree_digest"
