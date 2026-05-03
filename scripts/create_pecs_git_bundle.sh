#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <output.bundle> [ref]" >&2
  exit 1
fi

output_bundle="$1"
ref="${2:-HEAD}"

git rev-parse --is-inside-work-tree >/dev/null

mkdir -p "$(dirname "${output_bundle}")"

echo "[pecs-bundle] creating bundle for ${ref} -> ${output_bundle}"
git bundle create "${output_bundle}" "${ref}"
git bundle verify "${output_bundle}"

echo "[pecs-bundle] created ${output_bundle}"
echo "[pecs-bundle] remote clone example:"
echo "  git clone ${output_bundle} /workspace/vllm-pecs"
