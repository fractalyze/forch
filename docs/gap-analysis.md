# Where the opcode path loses to the handwritten NTT — and why

Measured 2026-08-21, RTX 5090 (1,792 GB/s GDDR7, 96 MB L2, 170 SMs),
`d = 2^16`, `q` = 60-bit CKKS prime (`2^60 − 2^18 + 1`), warm, 30 reps.
Handwritten = `benchmarks/handwritten/ntt.cu` (Shoup + Harvey lazy + ψ^brev,
two kernels, warp-per-256-pt). Opcode = `frx.lax.ntt` (NEGACYCLIC), the
generated `ntt_pass` fusions. Exact tables in `benchmarks/results/`.

| path (batch 64, µs/NTT, quiet GPU) | fwd | inv |
|---|---|---|
| handwritten | **1.35** | **1.37** |
| `lax.ntt` raw (natural order out) | 1.64 | 1.63 |
| `lax.ntt` + `lax.bit_reverse` (contract order) | 2.15 | — |
| `RnsRing.ntt` today (`fnp.take` adapter) | 2.23 | — |
| `Poly` product on a 25-limb ring, amortized | **12.7** | — |

Roofline: the two-kernel design moves 2 MB/NTT of DRAM traffic (the L2 does
NOT retain the 32 MB intermediate at batch 64 — apparent per-kernel bandwidth
never exceeds DRAM peak), so its floor is 1.12 µs/NTT. The handwritten
kernels sit at 97% (contiguous phase) and 79% (strided phase) of peak.

## Update 2026-08-22: most of the gap was scheduling, not codegen

The per-limb dispatch cost below is real, but its cause was not what this
document first claimed. XLA's default command-buffer mode (`LHS`) makes each
command depend on the previous one, so the 75 independent NTTs of a 25-limb
product ran strictly in sequence. With
`--xla_gpu_command_buffer_scheduling_mode=CONCURRENT`:

| | 25-limb product | per NTT | peak memory |
|---|---|---|---|
| default (`LHS`) | 0.96 ms | 12.8 µs | 51 MiB |
| `CONCURRENT` | 0.20 ms | 2.6 µs | 114 MiB (2.2×) |
| limb-batched handwritten (ceiling) | ~0.13 ms | 1.78 µs | — |

So ~5× of the original ~10× was a scheduling flag, and **~2.4× remained** —
that residue is the limb axis proper (fewer, larger launches). It was the
target of fractalyze/xla#569, which is now **merged and measured**; see the
update below. The flag is not proposed as an XLA default: it costs 2.2× peak memory because buffer assignment falls back to
`DependencyHloOrdering` and stops packing temporaries. Upstream defaulted it
on in July 2025 and reverted the same day.

## Update 2026-08-22 (later): the limb axis is closed — the glue is not

fractalyze/xla#569 landed (`f7e9504`): the compiler now merges an RNS
ciphertext's independent same-geometry NTT passes into one multi-root dispatch
of k grid-z planes, one monomorphic transform body per limb. It shipped gated
at `xla_gpu_ntt_max_fusion_group=1`; **fractalyze/xla#579 (`4e52227`) then made
grouping the default**, after measuring that it never fires on the zk provers'
shapes and that module PTX falls as the group grows rather than rising. So this
needs an frx carrying that xla and nothing more — `ntt_bench.py` still sends
the value (`FORCH_NTT_LIMB_GROUP`) to pin what it measured and to allow the
off-switch A/B, not to enable it.

Same card and parameters as above, `nsys --cuda-graph-trace=node`, GPU-busy =
the union of kernel intervals per iteration. Kernel-sum is the wrong statistic
for the ungrouped row — its 50 kernels overlap heavily under CONCURRENT (172 µs
of kernel time inside 45.6 µs of busy time), which is exactly what that flag
buys.

Forward pass, 25 transforms:

| | kernels | GPU busy | µs/NTT |
|---|---|---|---|
| before (per-limb dispatch) | 50 | 45.63 µs | 1.825 |
| grouped (`limbs=25`) | 2 | 26.14 µs | 1.045 |
| batched `lax.ntt` `[25, 2^16]`, one modulus | 2 | 26.01 µs | 1.041 |

**The generated kernel lands on the single-modulus batched bar to within
0.4%.** Given the limbs in one call, codegen was never the problem — the whole
residue was launch count.

The 25-limb product, 75 transforms, split by kernel:

| | NTT busy | elementwise busy | total | #ntt | #other |
|---|---|---|---|---|---|
| before | 215.43 µs | 174.78 | 215.59 µs | 150 | 100 |
| grouped | 132.80 µs | 70.74 | 179.81 µs | **6** | 100 |
| batched, one modulus | 100.79 µs | 6.65 | 107.38 µs | 6 | 1 |

**Delta 1 below is resolved.** Its issue-draft acceptance criterion — "25-limb
product within 1.5× of the equal-traffic single-modulus batched call" — is met
by the transforms at **1.32×** (150 fusions → 6, 215.4 → 132.8 µs, 1.62×).
Byte-identical per limb, verified against the ungrouped path on the real
58/42-bit prime set.

**What it exposed.** The product as a whole only improves 1.20× and sits at
1.67× the bar, because the residue is no longer the NTT: it is the other 100
kernels — the per-limb *pointwise multiply*, 70.74 µs against the batched
reference's 6.65 µs for one fused elementwise op. That is this document's
thesis restated one op over: a ring element carries one dtype per limb, so
every elementwise op over it is one kernel per limb too. It is filed as
measured context on xla#168 (NTT is a hard fusion boundary — adjacent
elementwise won't fuse in), whose `pre=`/`post=` boundary-op fold is the
mechanism that should absorb them. Two things block that today: the fold is
not firing for this shape at all (the ungrouped row already shows 100 separate
elementwise kernels), and it cannot compose with grouping — the fold's scale
operands sit at fixed positions that the per-limb `[k data, k twiddles]`
layout displaces, so the parser rejects the combination.

**Next largest item on this page is now delta 2**, the order adapter.

## The deltas, largest first

### 1. Per-limb dispatch: ~13 µs vs 1.33 µs amortized (~10×) — the real gap

> **Resolved 2026-08-22** by fractalyze/xla#569 (`f7e9504`); the numbers below
> are the diagnosis that motivated it, kept because the reasoning still holds.
> The shipped mechanism differs from the ask — grouping already-lowered
> per-limb fusions, not a new multi-modulus op. See the later update above.

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
runs. This was the highest-value item on the list.

### 2. Order adapter: 2.23 → 1.35 µs (1.65×) against the contract order

The FHE convention keeps the NTT domain bit-reversed (every CPU and GPU FHE
library surveyed; lattice-frx's contract order IS lattigo's bit-reversed
table order). The opcode emits natural order, so `RnsRing.ntt` pays a full
gather (`fnp.take`, +0.6 µs quiet / +1.35 µs contended) or, best case,
a `lax.bit_reverse` kernel (+0.5 µs) that the rewriter's DIF fold cannot
elide because `NEGACYCLIC_*` is pinned to CT-DIT
(`ntt_fusion_rewriter.cc:748-790` recognizes a `kBitReverse` consumer only
for the cyclic types).

**Ask (xla):** a bit-reversed-output mode for `NEGACYCLIC_NTT` (and
-input for `NEGACYCLIC_INTT`) — CT-DIT with a ψ^brev-indexed table emits it
natively, exactly as the handwritten kernel does; no permutation anywhere.
**Ask (lattice-frx):** until then, `RnsRing.ntt/intt` should use
`lax.bit_reverse` instead of `fnp.take` (−0.5–0.6 µs/NTT measured, and it
becomes a no-op the day the opcode grows the native mode).

### 3. Butterfly arithmetic: raw 1.64 vs 1.35 µs (~18%)

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

Self-contained per playbook §11. Struck-through items have shipped; the
rest are still drafts pending the owner's go-ahead.

1. ~~**xla: batch the RNS limb axis through one NTT call**~~ — **DONE.** Filed
   as fractalyze/xla#569, merged as `f7e9504`. The shipped form is not the
   sketch here (no new stacked type, no `[limbs, ..., d]` shape): the rewriter
   groups the already-lowered per-limb `ntt_pass` fusions into one multi-root
   fusion and the emitter monomorphizes a body per limb behind a grid-z switch,
   which needs no frontend or type-system change at all. Acceptance criterion
   met at 1.32× (bar: 1.5×) — see the 2026-08-22 update above.
2. **xla: native bit-reversed order for the negacyclic NTT** — problem: the
   FHE/lattigo contract order costs a gather (2.23 µs total) or an extra
   kernel (2.15 µs) against 1.64 µs raw; CT-DIT with ψ^brev tables emits
   bit-reversed for free. Acceptance: `RnsRing.ntt`-shaped call ==
   handwritten order with no permutation op in the HLO.
3. **lattice-frx: `fnp.take` → `lax.bit_reverse` in `RnsRing.ntt/intt`** —
   measured −0.1–0.6 µs/NTT depending on contention; forward-compatible with (2).
4. **(stretch, xla) Shoup/lazy butterflies for ≤60-bit parametric fields** —
   ~20% on the raw transform; only worth scheduling after (1) and (2), which
   dominate.
