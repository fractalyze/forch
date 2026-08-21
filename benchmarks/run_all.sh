#!/usr/bin/env bash
# One quiet-GPU pass of the whole table -> benchmarks/results/<date>-rtx5090/.
# Refuses to run while another process holds significant VRAM.
set -euo pipefail
cd "$(dirname "$0")/.."
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 2000 ]; then
  echo "GPU busy (${used} MiB in use) — refusing to record official numbers." >&2
  exit 1
fi
R="benchmarks/results/$(date +%F)-rtx5090"
mkdir -p "$R"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$R/machine.txt"
( cd benchmarks/handwritten && make -s ntt_bench && ./ntt_bench golden/q60 check >/dev/null && make -s bench ) > "$R/handwritten.txt"
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python benchmarks/ntt_bench.py 2>/dev/null > "$R/opcode.txt"
echo "wrote $R"
