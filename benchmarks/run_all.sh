#!/usr/bin/env bash
# One quiet-GPU pass of the whole table -> benchmarks/results/<date>-<gpu>/.
# Refuses to run while another process holds significant VRAM.
set -euo pipefail
cd "$(dirname "$0")/.."
# One line per GPU; gate on the busiest one.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "$used" -gt 2000 ]; then
  echo "GPU busy (${used} MiB in use) — refusing to record official numbers." >&2
  exit 1
fi
gpu_slug=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
  | tr '[:upper:] ' '[:lower:]-' | sed 's/nvidia-//;s/geforce-//')
R="benchmarks/results/$(date +%F)-${gpu_slug}"
mkdir -p "$R"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$R/machine.txt"
# Byte-exactness gates ALL cases before any number is recorded.
make -s -C benchmarks/handwritten ntt_bench check
( cd benchmarks/handwritten && make -s bench ) > "$R/handwritten.txt"
# Both scheduling modes: the opcode path's headline number depends on which one
# is in force (~5x), so recording one without the other is not a measurement.
for mode in CONCURRENT LHS; do
  FORCH_COMMAND_BUFFER_MODE="$mode" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python benchmarks/ntt_bench.py \
    > "$R/opcode-$mode.txt" 2> "$R/opcode-$mode.stderr.log" \
    || { cat "$R/opcode-$mode.stderr.log" >&2; exit 1; }
done
echo "wrote $R"
