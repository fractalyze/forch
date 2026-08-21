# forch v0 — polynomial multiplication that compiles to a fast NTT

Date: 2026-08-21 (rev 2 — the handwritten kernel demoted from product path to
reference baseline, per review). Status: approved direction; this is the
design to implement.

## 1. Purpose

forch is the FHE *evaluation* layer planned above `lattice-frx` ("PyTorch for
FHE"). Its v0 makes one point, the same point the rest of the zorch stack
makes about ZK kernels: **you write plain Python, it compiles through the
`ntt` opcode's GPU codegen, and that generated code is (to be made) as fast
as a handwritten FHE kernel.**

Three deliverables, one repo:

1. **The API**: a `Poly` you can just multiply — `a * b` inserts the
   NTT/domain moves itself, everything traces into one XLA graph. No tables,
   no FFI, no transform calls in user code.
2. **The reference**: a minimal handwritten CUDA NTT (the FHE literature's
   design: Shoup + Harvey lazy + ψ^brev tables, two kernels) as a standalone
   benchmark binary — the "this is what hand-tuned looks like, and this is
   its speed" bar. It is *not* wired into the product path.
3. **The comparison**: one table on one RTX 5090 — handwritten vs the opcode
   path (raw and in contract order) vs roofline vs published numbers — plus
   a gap analysis filed as `fractalyze/xla` issues (Shoup/lazy butterflies,
   native bit-reversed emission). Closing the gap is follow-up work in xla,
   not in this repo.

Out of scope for v0: CKKS itself (encode/encrypt/rescale/key-switch), any
xla emitter change, Pallas variants, 32-bit RNS, tensor cores, ring degrees
other than `2^16` in the benchmark (the API inherits whatever `RnsRing`
accepts).

## 2. Where it sits

```
hash-frx   lattice-frx  <- substrates
    \        /    \
   enc-frx ------ forch  <- function layers (enc-frx: secret keys; forch: eval only)
```

Dependencies: `lattice-frx` (ring, primes, roots, exact host oracle), `frx`
+ `frx-cuda12-plugin`, `zk-dtypes`, `numpy`. No dependency on `xla` source,
no custom calls, no scheme repos. The handwritten reference builds with
plain `nvcc` and links nothing from the stack.

## 3. The API (`forch/`)

```python
import forch
ring = forch.Ring(q_moduli, d=1 << 16)      # thin wrapper over lattice_frx.ring.RnsRing
a = ring.from_signed(coeffs)                 # or ring.poly(host_u64[limbs, d])
c = a * b                                    # NTT both sides, pointwise mul — automatic
d = c + a                                    # domain coercion automatic
d.coeffs()                                   # -> host [limbs, d], canonical residues
```

- `Poly` carries `Coeff` and/or `Eval` from `lattice_frx.ring`, converting
  lazily and caching both forms. Domain bookkeeping happens in Python at
  trace time — consistent with lattice-frx's "the domain is a type, not a
  runtime flag": forch only decides *which* statically-typed op to emit.
- `*` computes in Eval (`ring.mul`), `+`/`-` in whichever domain both
  operands already share (preferring the one that inserts no transform);
  `mul_add` fuses where the expression allows.
- Everything composes under `frx.jit`; the README shows the resulting HLO
  contains `ntt` fusions and the whole `a * b + c` is one compiled zone.
- Correctness oracle: `HostRnsRing` (negacyclic schoolbook on exact ints via
  its own `ntt`; plus an `O(d²)` naive negacyclic mul at small `d` to keep
  the oracle honest).

## 4. The handwritten reference (`benchmarks/handwritten/`)

Standalone CUDA17 binary (`Makefile`, `nvcc -arch=sm_120`), no Python, no
FFI. Design per `study/fhe/ntt/techniques.md`:

- **Arithmetic**: Shoup multiply (`w' = ⌊w·2^64/q⌋`, `__umul64hi`) with
  Harvey lazy ranges — forward CT butterflies in `[0, 4q)`, inverse GS in
  `[0, 2q)`, one full reduction at the store; `q < 2^60`.
- **Tables**: `T[brev(i)] = ψ^i` and Shoup companions, ψ from the same
  primitive-root walk as `lattice_frx.roots` (lattigo's), so values match
  the oracle byte-exactly, twist merged, order native (CT forward: natural →
  bit-reversed = the lattice-frx contract order; GS inverse: back).
- **Structure**: `2^16 = 2^8 × 2^8`, two kernels. Phase 1: a block loads a
  `256 × T_cols` strided tile coalesced into padded shared memory, one warp
  per column runs a 256-pt transform with 8 elements/lane in registers
  (radix-8 → exchange → radix-8 → exchange → radix-4, `__syncwarp` only),
  twiddles `T[m'+i']`. Phase 2: contiguous chunks, same warp routine,
  twiddles `T[m'·(256+c)+i']` — the split is twist-free with this table
  (verified against `HostRnsRing` numerically; the check ships as a test).
  `T_cols ∈ {8,16,32}` and chunks-per-block `W ∈ {4,8}` are compile-time
  sweep parameters.
- **Harness**: golden vectors dumped by a small Python script
  (`make_golden.py`, uses `HostRnsRing`) into flat files; the binary loads
  them, checks byte-exact forward/inverse/round-trip, then times with CUDA
  events: batch ∈ {1, 16, 64, 256}, primes {`2^60−2^18+1`,
  `FirstPrime(50)`, `LastPrime(58)`, `LastPrime(42)`} — the OpenFHE-default
  and HEaaN-FGb shapes.
- Roofline on 1,792 GB/s: 0.585 µs/NTT (intermediate in L2) – 1.17 µs
  (spilled); literature scaled from Ada: 0.9–1.2 µs. If the handwritten
  kernel can't beat the opcode's 1.42 µs it is not a bar, and the design is
  revisited before any number is quoted.

## 5. Verification

`absltest`, `forch/testing/`:

1. `(a * b).coeffs() == ` exact negacyclic product from `HostRnsRing`
   (its NTT path), random inputs, primes 60/50/58/42-bit, batch 1 and 3;
   plus `O(d²)` naive check at `d = 64`.
2. Domain bookkeeping: `+` after `*` inserts no extra transform (count `ntt`
   ops in the jaxpr/HLO text); explicit `.ntt()`/`.intt()` round-trip.
3. `Poly` under `frx.jit`: one compiled call, same bytes as eager.
4. Handwritten binary: golden forward/inverse/round-trip gates run by
   `make check` (CUDA-capable CI/host only), including the all-`q−1` lazy
   overflow probe and canonical (`< q`) outputs.
5. The pure-Python two-phase split check (CPU, pins the §4 algebra).

## 6. Benchmark and report (`benchmarks/`, `README.md`)

One table, same card, same day: handwritten fwd/inv · `lax.ntt` raw ·
`lax.ntt` + `lax.bit_reverse` · `RnsRing.ntt` (today's `fnp.take` path) ·
`Poly.__mul__` end-to-end (2×NTT + pointwise, per-NTT amortized) — µs/NTT
and effective GB/s, batch and prime sweeps as in §4, one `nsys` capture.
Already measured for context: opcode raw 1.42 µs, contract order 2.77 µs.

README also carries: an annotated walkthrough of the handwritten kernel
(the butterfly, the table indexing, the two phases) beside the HLO the
opcode path generates for `a * b` — the "what does the code look like" half
of the deliverable — and the gap analysis: which of {Shoup, lazy ranges,
native brev order, table-vs-window twiddles} the emitter lacks, each filed
as a `fractalyze/xla` issue with the measured delta it should recover, and
the `lattice-frx` `fnp.take` → `lax.bit_reverse` issue (−0.6 µs measured).

## 7. Repository layout

```
forch/
  forch/__init__.py            # Ring, Poly
  forch/testing/poly_test.py, split_test.py
  benchmarks/handwritten/ntt.cu, harness.cu, Makefile, make_golden.py
  benchmarks/ntt_bench.py      # opcode-path + Poly timings
  README.md, CLAUDE.md, LICENSE (Apache-2.0), pyproject.toml
  docs/superpowers/specs/…
```

Pure-Python package (`pip install -e .` — no build step); the handwritten
reference builds only where CUDA exists. Bazel wiring: follow-up.

## 8. Risks / open edges

- 8 elements/lane at 64-bit may spill (FIDESlib's reason for radix-2):
  `-Xptxas -v` gate; fallback radix-4/lane, same structure.
- Phase-1 strided tile may need `T_cols = 32` (66 KB smem, 1 block/SM) for
  bandwidth — the sweep decides, occupancy noted in the report.
- Batch 1 under-fills 170 SMs on both paths — reported as-is; batch ≥ 16 is
  the honest CKKS number (a ciphertext is dozens of limbs).
- `Poly` fusion behavior (does XLA fuse pointwise mul into the NTT store?) is
  observed and reported, not engineered, in v0.
