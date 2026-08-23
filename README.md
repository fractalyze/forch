# forch

The FHE evaluation layer on the [lattice-frx](https://github.com/fractalyze/lattice-frx)
substrate — "PyTorch for FHE". You write the polynomial arithmetic; the
compiler does the NTT.

```python
import forch
ring = forch.Ring(q_moduli, d=1 << 16)   # CKKS-scale: q_i < 2^60, q_i ≡ 1 mod 2^17
c = a * b                                 # NTT both sides, pointwise mul — automatic
(c + e).coeffs()                          # domain moves inserted at trace time
```

`Poly` tracks its domain (coefficient vs NTT) in Python at trace time and
inserts the transforms itself — `a * b + c` compiles to one XLA graph whose
transforms are `ntt` opcode fusions. Everything byte-matches lattice-frx's
exact host ring, in its (lattigo's) bit-reversed contract order. v0 is this
API plus one benchmark; the CKKS layers (mult with relinearization,
rotation, rescale) come next and consume exactly these pieces.

## The question v0 answers

**How fast is a CKKS-parameter NTT (`d = 2^16`, 40–60-bit RNS primes) on an
RTX 5090 — and how close does the `ntt` opcode's generated code get to a
handwritten FHE kernel?** No public RTX 5090 NTT microbenchmark existed at
the time of writing.

`benchmarks/handwritten/ntt.cu` is the reference bar: the FHE literature's
design — Shoup multiplication with precomputed `⌊w·2^64/q⌋` companions and
Harvey lazy ranges (`[0,4q)` forward / `[0,2q)` inverse; Harvey JSC'14,
Longa–Naehrig CANS'16), ψ^brev twiddle tables (negacyclic twist merged, no
bit-reversal ever materialized — the same tables OpenFHE/SEAL/HEXL/Lattigo/
Phantom build), split `2^16 = 2^8 × 2^8` into two kernels, each 256-point
sub-transform done by one warp holding 8 elements per lane (3 register
stages + 5 shuffle stages, `__syncwarp` only). It is a standalone `nvcc`
binary, byte-gated against the host ring — a benchmark reference, **not** a
product path: the plan is to close the gap in the compiler, not to wire
custom kernels around it.

## Results (RTX 5090, d = 2^16, 60-bit prime, µs per NTT, warm)

<!-- BENCH_TABLE_START: benchmarks/results/2026-08-21-rtx5090 (quiet GPU) -->
| path | batch 1 | 16 | 64 | 256 |
|---|---|---|---|---|
| handwritten fwd | 21.2 | 1.78 | **1.35** | 1.54 |
| handwritten inv | 20.7 | 1.78 | **1.37** | 1.55 |
| `lax.ntt` raw fwd (natural order) | 20.5 | 2.53 | 1.64 | 1.96 |
| `lax.ntt` raw inv | 17.9 | 2.22 | 1.63 | 1.67 |
| `lax.ntt` + `lax.bit_reverse` (contract order) | 20.8 | 2.71 | 2.15 | 2.91 |
| `RnsRing.ntt` today (`fnp.take` adapter) | 19.4 | 2.88 | 2.23 | 2.30 |
<!-- BENCH_TABLE_END -->

- Roofline: the two-kernel design moves 2 MB of DRAM per NTT (measured: the
  96 MB L2 does *not* retain the 32 MB intermediate at batch 64) → floor
  1.12 µs. The handwritten kernels run at 97% (contiguous phase) / 79%
  (strided phase) of the card's 1,792 GB/s.
- Batch 1 is launch/latency-bound on every path (256 blocks cannot fill
  170 SMs) — batch ≥ 16 is the honest CKKS regime, since a ciphertext is
  dozens of limbs. Published context: Phantom ≈1.5 µs/limb and GPU-NTT
  8.7 µs single on an RTX 4090 (1,008 GB/s).
- A 25-limb (HEaaN-FGb-shaped) `Poly` product amortizes to **12.8 µs/NTT**
  under XLA's default scheduling and **2.6 µs/NTT** with concurrent command
  buffers (`benchmarks/ntt_bench.py` sets that itself — see below). The gap is
  the *number of kernel launches*, not butterfly arithmetic: per-limb dtypes
  force one batch-1 transform per limb, and XLA's default command-buffer mode
  serializes them even though nothing connects them. Full attribution in
  [docs/gap-analysis.md](docs/gap-analysis.md).
- **The transform half of that is now closed upstream.**
  [fractalyze/xla#569](https://github.com/fractalyze/xla/issues/569) merged the
  limb axis into one dispatch: 150 transform dispatches → 6, NTT device time
  215.4 → 132.8 µs, landing within 0.4% of an equal-traffic single-modulus
  batched call — given the limbs in one call, the generated kernel *matches*
  the batched one. What remains is the per-limb **pointwise** ops, the same
  one-dtype-per-limb problem one op over. Needs an frx carrying that xla —
  [xla#579](https://github.com/fractalyze/xla/pull/579) then made grouping the
  backend default, so nothing has to ask for it; the bench prints which state
  it got.

### One flag moves the headline number 5×

XLA's default command-buffer mode (`LHS`) makes every command depend on the
previous one, so 75 independent NTTs run strictly in sequence. Setting
`--xla_gpu_command_buffer_scheduling_mode=CONCURRENT` lets the CUDA graph's
buffer-conflict DAG overlap them:

| | 25-limb product | per NTT | peak memory |
|---|---|---|---|
| default (`LHS`) | 0.96 ms | 12.8 µs | 51 MiB |
| `CONCURRENT` | **0.20 ms** | **2.6 µs** | 114 MiB (2.2×) |

`benchmarks/ntt_bench.py` sets it in-process rather than documenting it, so a
forgotten environment variable cannot publish a 5× regression as a
measurement; `FORCH_COMMAND_BUFFER_MODE=LHS` overrides it, and
`benchmarks/run_all.sh` records both. It is deliberately *not* proposed as an
XLA default: it costs ~2.2× peak memory (buffer assignment stops packing
temporaries), which is free at FHE sizes and would not be on a memory-bound
prover. Upstream tried defaulting it on and reverted the same day.

## The code, side by side

The handwritten butterfly (`benchmarks/handwritten/ntt.cu`):

```cuda
__device__ u64 mul_shoup(u64 x, u64 w, u64 ws, u64 q) {
  return x * w - __umul64hi(x, ws) * q;          // [0, 2q), any x
}
// CT stage, values lazy in [0, 4q):
if (x >= two_q) x -= two_q;
u64 t = mul_shoup(y, w, ws, q);                  // w = psi^brev-table twiddle
y = x + two_q - t;
x = x + t;
```

The opcode path for the same ring op is ordinary traced Python — the whole
`a * b` lowers to `ntt_pass` custom fusions you can read back:

```python
print(frx.jit(lambda a, b: (Poly(ring, coeff=a) * Poly(ring, coeff=b))
              .as_coeff().limbs).lower(ca, cb).compile().as_text())
```

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev,cuda]' \
    --extra-index-url=https://fractalyze.github.io/pypi/simple/
.venv/bin/python -m forch.testing.poly_test     # oracle byte-match (CPU ok)
.venv/bin/python -m forch.testing.trace_test    # transform-count guarantees
.venv/bin/python -m forch.testing.split_test    # the 2^8x2^8 algebra, exact ints

# The handwritten reference (CUDA 12.9, sm_120):
cd benchmarks/handwritten
make golden      # dump vectors + tables from the host ring
make check       # byte-exact fwd / inv / round-trip gates, 4 primes
make bench       # CUDA-event sweep
cd ../.. && bash benchmarks/run_all.sh          # the whole table, quiet GPU only
```

## Parameters

`q ≡ 1 (mod 2^17)` so the 2d-th root ψ exists. The benchmark set covers the
OpenFHE default shape (60-bit first modulus + 50-bit scaling primes) and the
CryptoLab HEaaN FGb shape (58 + 42 bit, log QP = 1555):

| case | q | provenance |
|---|---|---|
| q60 | `1152921504606584833` = 2^60−2^18+1 | OpenFHE `LastPrime(60, 2^17)`; also SEAL/HEXL/Lattigo's |
| q50 | `1125899908022273` | OpenFHE `FirstPrime(50, 2^17)` (first scaling prime) |
| q58 | `288230376147386369` | `LastPrime(58, 2^17)`, HEaaN FGb base-prime shape |
| q42 | `4398044938241` | `LastPrime(42, 2^17)`, HEaaN FGb quantize-prime shape |

(The familiar `2^60 − 2^14 + 1` is *not* here on purpose: its 2-adicity is
14, so it has no 2^17-th root — it only works for d ≤ 2^13.)

## Layout

```
forch/                      Ring, Poly (the product path — pure Python, traced)
forch/testing/              absltest suites: oracle byte-match, trace counts, split algebra
benchmarks/handwritten/     the reference CUDA kernel + golden harness (not a product path)
benchmarks/ntt_bench.py     opcode-path timings; run_all.sh for the full table
docs/gap-analysis.md        where codegen loses today, as filed-issue drafts
```
