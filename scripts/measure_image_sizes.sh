#!/usr/bin/env bash
# Build Dockerfile.baseline (full requirements.txt, default PyPI torch, pip cache kept)
# and Dockerfile (requirements-prod.txt, CPU-only torch, --no-cache-dir) and print a
# markdown before/after table.  Run on YOUR machine or via the image-size-report
# GitHub workflow - NOT in the restricted lab workspace.
set -euo pipefail
cd "$(dirname "$0")/.."

docker build -q -t finbert-before -f Dockerfile.baseline . >/dev/null
before=$(docker image inspect -f '{{.Size}}' finbert-before)
docker rmi -f finbert-before >/dev/null   # keep disk usage low on CI runners
docker build -q -t finbert-after -f Dockerfile . >/dev/null
after=$(docker image inspect -f '{{.Size}}' finbert-after)

python3 - "$before" "$after" <<'PY'
import sys
b, a = int(sys.argv[1]), int(sys.argv[2])
gb = lambda x: f"{x / 1e9:.2f} GB"
print("| Image | Dockerfile | Size |")
print("| --- | --- | --- |")
print(f"| before | Dockerfile.baseline | {gb(b)} |")
print(f"| after  | Dockerfile | {gb(a)} |")
print(f"\nReduction: {gb(b - a)} ({100 * (b - a) / b:.1f}%)")
PY
