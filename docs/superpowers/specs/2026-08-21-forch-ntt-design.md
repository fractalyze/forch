# forch — negacyclic NTT for CKKS-scale rings on GPU (v0 design)

Date: 2026-08-21. Status: approved approach (out-of-tree CUDA FFI, NTT-only
scope, OpenFHE 60/50 + HEaaN 58/42 prime sets); this document is the design
to implement.

## 1. Purpose

forch is the FHE *evaluation* layer planned above `lattice-frx` ("PyTorch for
FHE": mult / relinearization / rotation / bootstrapping over ciphertexts —
nothing that touches a secret key). Its first and smallest deliverable is the
operation every one of those is made of: the negacyclic NTT over
`Z_q[X]/(X^N + 1)` at CKKS scale (`N = 2^16`, `q < 2^60`), written the way
the FHE literature writes it, running on an RTX 5090, with a number and a
readable kernel to show for it.

v0 answers one question: **how fast is a CKKS-parameter NTT on an RTX 5090
when the known FHE tricks are applied, and what does that code look like** —
measured against the stack's existing `frx.lax.ntt` on the same card in the
same process.

Out of scope for v0: CKKS itself (encoder, encrypt, mult, rescale,
key-switch), any change to `lattice-frx` or `xla`, Pallas variants, 32-bit
RNS (Cheddar-style), tensor-core NTTs, ring degrees other than `2^16`.

## 2. Where it sits

```
hash-frx   lattice-frx  <- substrates
    \        /    \
   enc-frx ------ forch  <- function layers (enc-frx: keys; forch: eval keys only)
```

forch depends on `lattice-frx` (ring types, primes, roots, the exact host
ring as oracle), `frx` + `frx-cuda12-plugin` (arrays, `frx.ffi`), `zk-dtypes`
(field dtypes), `numpy`. It never depends on `enc-frx`/`sig-frx`, and it does
not reach into `xla` — the kernel is an out-of-tree shared object registered
through `frx.ffi`, so no XLA build is involved.

## 3. Contract

```python
ring = lattice_frx.ring.RnsRing(q_moduli, d=1 << 16)     # limbs q_i ≡ 1 mod 2^17, q_i < 2^60
plan = forch.ntt.NttPlan(ring)                           # builds + uploads per-limb tables once
eval_ = plan.ntt(coeff)     # Coeff -> Eval, lattigo (bit-reversed) order, == HostRnsRing.ntt
coeff = plan.intt(eval_)    # Eval  -> Coeff, includes 1/N,                  == HostRnsRing.intt
```

- Input/output types are `lattice_frx.ring.Coeff` / `Eval` (tuple of per-limb
  field arrays, `[..., d]`, leading axes are batch). Same values, same order,
  same `1/d` convention as `RnsRing.ntt/intt` and `HostRnsRing`. The order
  comes out of the kernel natively (CT-DIT forward emits bit-reversed order;
  GS-DIF inverse consumes it) — there is no permutation anywhere.
- Limb arrays are handed to the kernel by `lax.bitcast_convert_type` to raw
  words and back. The field dtype's storage may be Montgomery; the kernel
  does not care, because every multiplication in the transform is "data ×
  constant twiddle" (Shoup), which is representation-agnostic: `NTT(x·R) =
  NTT(x)·R`. Twiddle tables are plain residues.
- Output residues are fully reduced to `[0, q)` (the canonical contract of
  `lattice-frx`); the lazy `[0, 4q)` range exists only inside the kernel.
- Supported: `d == 2^16` exactly, `q < 2^60`, `q ≡ 1 (mod 2^17)`. Anything
  else raises `ValueError` at `NttPlan` construction, naming the bound. The
  60-bit cap is Harvey's `4q < 2^62` headroom plus the same margin every FHE
  library keeps (SEAL/OpenFHE/Phantom/HEonGPU cap at 60).
- `d` other than `2^16` is a deliberate v0 restriction, not a design limit:
  the split below generalises to `N1 = 2^8, N2 = N/2^8`.

## 4. Kernel design

Per limb, per call: two CUDA kernels, forward or inverse, over a flattened
`[batch, 65536]` u64 buffer. The `2^16`-point transform is decomposed as
`2^8 × 2^8`; each `2^8`-point sub-transform is done by **one warp holding 8
elements per lane in registers**, so the only synchronisation inside a
sub-transform is `__syncwarp()`.

### 4.1 Arithmetic — Shoup multiply, Harvey lazy butterflies

Precomputed per twiddle `w`: `w' = ⌊w · 2^64 / q⌋`.

```
mul_shoup(x, w, w'):  hi = __umul64hi(x, w');  r = x*w - hi*q;   // r ∈ [0, 2q)
CT (forward):  if (x ≥ 2q) x -= 2q;  t = mul_shoup(y, w, w');  x' = x + t;  y' = x - t + 2q;   // in/out [0, 4q)
GS (inverse):  u = x + y; if (u ≥ 2q) u -= 2q;  v = mul_shoup(x - y + 2q, w, w');               // in/out [0, 2q)
```

Forward input is `[0, q)`, kept in `[0, 4q)` across the 16 stages; the store
reduces with two conditional subtractions. Inverse keeps `[0, 2q)`. `N⁻¹` is
folded into the inverse's last-stage twiddle (the `x+y` half still takes one
`mul_shoup` by `N⁻¹`, as OpenFHE does after issue #872). Reference: Harvey,
JSC 2014 (arXiv 1205.2926); Longa–Naehrig, CANS 2016 (eprint 2016/504).

### 4.2 Tables — ψ^brev, twist merged, no bit-reversal

`ψ = g^((q−1)/2N)` with `g = lattice_frx.roots.primitive_root(q, …)` — the
same generator `RnsRing` hands `lax.ntt`, so the root (and therefore every
value) is identical to `HostRnsRing`, not merely "an" NTT.

Per limb, four `u64[N]` tables: `T[brev(i)] = ψ^i`, its Shoup companion,
`Tinv[brev(i)] = ψ^(−i)`, its Shoup companion (2 MB per limb, built on the
host with Python integers, uploaded once per `NttPlan`). Indexing the CT
butterflies of stage `s` (`m = 2^s`) at group `i` with `T[m + i]` merges the
negacyclic twist into the transform (Roy et al. CHES 2014; Pöppelmann–Oder–
Güneysu 2015), and is the same table OpenFHE
(`transformnat-impl.h:730-738`), SEAL, HEXL, Lattigo and Phantom build.

### 4.3 Two phases, twist-free (verified numerically against `HostRnsRing`)

With that table the `2^8 × 2^8` split needs **no inter-phase twiddle
multiply** — both phases are plain 256-point CT-DIT transforms over different
slices of the same table:

- **Phase 1** (stages 0–7): column `j ∈ [0, 256)` = elements `{j + 256·k}`,
  twiddle for local stage `m'`, group `i'` = `T[m' + i']` (identical for every
  column).
- **Phase 2** (stages 8–15): contiguous chunk `c ∈ [0, 256)` = elements
  `[256c, 256c + 256)`, twiddle = `T[m'·(256 + c) + i']`.

The inverse mirrors it (GS-DIF, phase order reversed, `Tinv`). Checked in
Python against `HostRnsRing.ntt` at `q = 2^60 − 2^18 + 1` before this spec
was written; the same check becomes a unit test.

### 4.4 Memory movement

- **Phase 1 kernel**: a block owns `T_cols` adjacent columns (`T_cols ∈ {8,
  16, 32}`, template parameter, tuned by the benchmark), i.e. a
  `256 × T_cols` tile whose rows are contiguous `8·T_cols` bytes in global
  memory — loaded coalesced into padded shared memory (row stride
  `T_cols + 1` words to break 64-bit bank conflicts), then warp `w` runs the
  256-point transform on column `w` from shared memory with 8 elements per
  lane, and the tile is stored back the same way. Block = `32 · T_cols`
  threads; shared = `256 · (T_cols + 1) · 8` bytes (`T_cols = 16` → 34 KB,
  512 threads, 2–3 blocks per sm_120 SM).
- **Phase 2 kernel**: a block owns `W` consecutive chunks (`W ∈ {4, 8}`), each
  warp loads its 256 contiguous elements directly (8 per lane, fully
  coalesced), transforms in registers + a small per-warp shared scratch for
  the two intra-warp exchanges, and stores in place.
- Inside a 256-point warp transform the 8 stages are: radix-8 in registers
  (3 stages) → shared exchange → radix-8 (3) → shared exchange → radix-4 (2),
  with `__syncwarp()` between — no block-wide barriers after the initial tile
  load.
- Grid = `batch × 256 / T_cols` (phase 1) and `batch × 256 / W` (phase 2)
  blocks; the batch axis is the only thing that fills the 170 SMs, which is
  why the benchmark sweeps it.
- DRAM traffic: 2 × (read + write) = 2 MB per NTT if the 512 KB intermediate
  is evicted, 1 MB if it stays in the 96 MB L2 (it will for batch ≤ 64).
  Twiddle traffic is 1 MB per limb per direction but L2-resident across the
  batch. Roofline on 1,792 GB/s: **0.585 µs (1 MB) – 1.17 µs (2 MB)** per
  NTT.

### 4.5 FFI binding

One XLA FFI handler per direction, registered under `platform="CUDA"` via
`frx.ffi.register_ffi_target`, bound with `Ffi::Bind().Ctx<PlatformStream<
cudaStream_t>>().Arg<Buffer>(x).Arg<Buffer>(table).Arg<Buffer>(shoup)
.Ret<Buffer>(y).Attr<uint64_t>("q")...`. Words travel as the dtype `frx`
exposes without x64 (`uint32[..., 2]` pairs); the handler reinterprets the
buffer as `u64`. The Python side wraps `frx.ffi.ffi_call(...)` with
`vmap_method="broadcast_all"` so leading axes batch. Built with CMake +
nanobind after `jax/examples/ffi` (scikit-build-core, `nvcc` for sm_120),
against `frx.ffi.include_dir()`.

## 5. Verification

`absltest` suite, `forch/testing/ntt_test.py`, GPU required:

1. `ntt == HostRnsRing.ntt` and `intt == HostRnsRing.intt`, byte-exact, random
   inputs, for each prime in the benchmark set (60, 50, 58, 42 bit), batch 1
   and batch 3.
2. `intt(ntt(x)) == x` and `ntt(intt(y)) == y`.
3. Agreement with `RnsRing.ntt` (the `lax.ntt` path) — the two GPU paths
   must coincide.
4. Outputs canonical: every residue `< q` (exercises the final reduction),
   on all-zero, all-`q−1`, and random inputs (the `q−1` case is the lazy
   overflow probe).
5. Montgomery-stored limbs (`RnsRing.coeff_from_host`) and `storage="std"`
   limbs both round-trip.
6. `NttPlan` rejects `q ≥ 2^60`, `q ≢ 1 mod 2^17`, and `d ≠ 2^16` with
   `ValueError` naming the rule.
7. The pure-Python 2-phase split check from §4.3 (CPU-only, keeps the
   algebra pinned independently of the kernel).

## 6. Benchmark and report

`benchmarks/ntt_bench.py`, one process, warm, `block_until_ready` timing
over ≥30 reps, reporting µs/NTT and effective GB/s on the 1 MB model:

- paths: forch forward / inverse; `lax.ntt` raw; `lax.ntt` + `lax.bit_reverse`;
  `RnsRing.ntt` (the `fnp.take` adapter) — all on the same arrays.
- sweep: batch ∈ {1, 16, 64, 256}; primes {`2^60−2^18+1`, `FirstPrime(50)`,
  `LastPrime(58)`, `LastPrime(42)`}; `T_cols` / `W` variants for forch.
- one `nsys` capture of the batch-64 case for per-kernel device time.
- the README carries the resulting table, the already-measured baseline
  (`lax.ntt` 1.42 µs raw / 2.77 µs in contract order), the roofline, and an
  annotated walkthrough of `ntt.cu` (butterfly, table indexing, the two
  phases) — the "what does the code look like" half of the deliverable.

Expected from the literature (Phantom 1.52 µs/limb on a 4090 at 1,008 GB/s,
scaled by bandwidth): **0.9–1.2 µs/NTT**. Anything above 1.4 µs means the
kernel lost to the generic one and the design is revisited before any
claim is made.

## 7. Repository layout

```
forch/
  forch/__init__.py
  forch/ntt/__init__.py        # NttPlan, ntt, intt
  forch/ntt/tables.py          # per-limb ψ^brev + Shoup tables (host, exact ints)
  forch/ntt/_ffi.py            # frx.ffi registration + ffi_call wrappers
  forch/ntt/kernels/ntt.cu     # the two kernels, both directions
  forch/ntt/kernels/ffi.cc     # XLA FFI handlers + nanobind module
  forch/testing/ntt_test.py
  benchmarks/ntt_bench.py
  CMakeLists.txt, pyproject.toml (scikit-build-core)
  README.md, CLAUDE.md, LICENSE (Apache-2.0)
  docs/superpowers/specs/…     # this file
```

Bazel wiring (the org's other repos build hermetically) is a follow-up once
the kernel and numbers exist; v0 installs with `pip install -e .` from the
same Fractalyze index `lattice-frx` uses.

## 8. Risks and what decides them

- **FFI word type**: if `frx.ffi` refuses `uint32[..., 2]` buffers or
  `bitcast_convert_type` is not free under `jit`, fall back to enabling x64
  inside forch only, after checking `lattice-frx`'s "no x64" assumption
  still holds for field arrays. First task of the plan is this round-trip.
- **8 elements/lane spills** at 64-bit (FIDESlib fell back to radix-2 for
  this reason): checked with `-Xptxas -v`; the fallback is radix-4 per lane
  (4 elements), same structure.
- **Phase-1 tile load dominates** (strided side): `T_cols` sweep; if 32-word
  rows are needed for bandwidth, shared memory goes to 66 KB and occupancy
  to one block per SM — the benchmark decides.
- **Occupancy vs. batch**: batch 1 is one limb = 256 blocks, under-filling
  170 SMs × several blocks — reported as is, batch ≥ 16 is the honest
  number for CKKS (a ciphertext has dozens of limbs).
