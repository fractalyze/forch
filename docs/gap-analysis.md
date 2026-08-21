# Where the opcode path loses to the handwritten NTT — and why

Measured 2026-08-21, RTX 5090 (1,792 GB/s GDDR7, 96 MB L2, 170 SMs),
`d = 2^16`, `q` = 60-bit CKKS prime (`2^60 − 2^18 + 1`), warm, 30 reps.
Handwritten = `benchmarks/handwritten/ntt.cu` (Shoup + Harvey lazy + ψ^brev,
two kernels, warp-per-256-pt). Opcode = `frx.lax.ntt` (NEGACYCLIC), the
generated `ntt_pass` fusions. Exact tables in `benchmarks/results/`.

| path (batch 64, µs/NTT) | fwd | inv |
|---|---|---|
| handwritten | **1.33** | **1.37** |
| `lax.ntt` raw (natural order out) | 1.6–1.8 | 1.7–1.8 |
| `lax.ntt` + `lax.bit_reverse` (contract order) | 2.4 | — |
| `RnsRing.ntt` today (`fnp.take` adapter) | 2.7 | — |
| `Poly` product on a 25-limb ring, amortized | **~13** | — |

Roofline: the two-kernel design moves 2 MB/NTT of DRAM traffic (the L2 does
NOT retain the 32 MB intermediate at batch 64 — apparent per-kernel bandwidth
never exceeds DRAM peak), so its floor is 1.12 µs/NTT. The handwritten
kernels sit at 97% (contiguous phase) and 79% (strided phase) of peak.

## The deltas, largest first

### 1. Per-limb dispatch: ~13 µs vs 1.33 µs amortized (~10×) — the real gap

`RnsRing` carries one dtype per limb (`prime_field(q_i)`), so a 25-limb
product issues 75 batch-1 `lax.ntt` calls; a batch-1 NTT is latency-bound at
~20 µs on both paths (256-block grids under-fill 170 SMs; Phantom sees the
same on A100). Every GPU FHE library batches the limb axis into one grid
(Phantom: `n/8 × limbs` threads; 100x: block = (limb, tile); Cheddar/FIDESlib
likewise) — limbs share nothing but `q`, ψ-table, and Shoup constants.

**Ask (xla):** a multi-modulus NTT — one `ntt` call over `[limbs, batch, d]`
with per-limb algebraic types (or a stacked-modulus type), lowering to one
fused `ntt_pass` chain whose twiddle constant carries all limbs' tables.
Recovers ~10× on ring-element products at CKKS limb counts, which is the
shape every consumer (jindo commit, future forch mult/key-switch) actually
runs. This is the highest-value item on the list.

### 2. Order adapter: 2.7 → 1.33 µs (2.1×) against the contract order

The FHE convention keeps the NTT domain bit-reversed (every CPU and GPU FHE
library surveyed; lattice-frx's contract order IS lattigo's bit-reversed
table order). The opcode emits natural order, so `RnsRing.ntt` pays a full
gather (`fnp.take`, +1.35 µs — as expensive as the transform) or, best case,
a `lax.bit_reverse` kernel (+0.6 µs) that the rewriter's DIF fold cannot
elide because `NEGACYCLIC_*` is pinned to CT-DIT
(`ntt_fusion_rewriter.cc:748-790` recognizes a `kBitReverse` consumer only
for the cyclic types).

**Ask (xla):** a bit-reversed-output mode for `NEGACYCLIC_NTT` (and
-input for `NEGACYCLIC_INTT`) — CT-DIT with a ψ^brev-indexed table emits it
natively, exactly as the handwritten kernel does; no permutation anywhere.
**Ask (lattice-frx):** until then, `RnsRing.ntt/intt` should use
`lax.bit_reverse` instead of `fnp.take` (−0.5–0.6 µs/NTT measured, and it
becomes a no-op the day the opcode grows the native mode).

### 3. Butterfly arithmetic: raw 1.6–1.8 vs 1.33 µs (~20%)

The generated kernel multiplies via Montgomery REDC (PrimeIR's choice for a
parametric 64-bit field) and fully reduces every butterfly; the handwritten
kernel uses Shoup multiplication (the twiddle is a constant — its
`⌊w·2^64/q⌋` companion is free) with Harvey lazy ranges (`[0,4q)` forward /
`[0,2q)` inverse, one reduction at the store), the design every FHE library
uses (Harvey JSC'14; SEAL/HEXL/OpenFHE/Phantom/FIDESlib). On a GPU whose
64-bit multiply is emulated on the 32-bit datapath, Shoup+lazy ≈ 3 fewer
wide multiplies + 2 fewer corrections per butterfly.

**Ask (xla):** in the NTT emitter, when the field is a parametric prime
≤ 60 bits, emit Shoup butterflies from a widened twiddle constant (table +
companions) with lazy ranges — the twiddle constant builder already owns the
table layout, so the companion row is one more precompute.

### 4. Not worth filing

- Twiddle supply (window-product vs flat table): both are L2-resident at
  these sizes; no measurable delta at d = 2^16.
- Phase-1 strided coalescing: the generated kernel already ports sppark's
  swizzled variant; the handwritten one is at 79% of peak there — the
  remaining sliver is not where the story is.

## Issue drafts

Drafts to file (pending owner's go-ahead), self-contained per playbook §11:

1. **xla: batch the RNS limb axis through one NTT call** — problem: per-limb
   dtypes force batch-1 NTTs; measured 20 µs/NTT batch-1 vs 1.3–1.8 µs
   batched on RTX 5090 at d=2^16; a 25-limb product amortizes to ~13 µs/NTT.
   Sketch: accept `[limbs, ..., d]` with a per-limb modulus list on the type
   or a new stacked type; twiddle constant becomes `[limbs, table]`; grid
   flattens `limbs × batch`. Acceptance: 25-limb product within 1.5× of the
   equal-traffic single-modulus batched call.
2. **xla: native bit-reversed order for the negacyclic NTT** — problem: the
   FHE/lattigo contract order costs a gather (2.7 µs total) or an extra
   kernel (2.4 µs) against 1.42 µs raw; CT-DIT with ψ^brev tables emits
   bit-reversed for free. Acceptance: `RnsRing.ntt`-shaped call ==
   handwritten order with no permutation op in the HLO.
3. **lattice-frx: `fnp.take` → `lax.bit_reverse` in `RnsRing.ntt/intt`** —
   measured −0.5–0.6 µs/NTT today; forward-compatible with (2).
4. **(stretch, xla) Shoup/lazy butterflies for ≤60-bit parametric fields** —
   ~20% on the raw transform; only worth scheduling after (1) and (2), which
   dominate.
