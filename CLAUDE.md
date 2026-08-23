# Project context for Claude Code

Read [README.md](README.md) first — the thesis (opcode path is the product,
the handwritten kernel is a benchmark reference), the measured table, and the
parameter set live there. Rules that gate changes:

- **Byte-exactness is the merge bar.** Every path — `Poly`, the opcode
  calls, the handwritten binary — must equal `lattice_frx.host_ring.
  HostRnsRing` exactly, in its (lattigo's) bit-reversed order, with ψ derived
  from `lattice_frx.roots.primitive_root`. Never introduce an independent
  root search; two valid NTTs with different ψ byte-disagree.
- **The handwritten kernel stays out of the product path.** No FFI, no
  custom calls, no wiring it under `forch/`. The gap it demonstrates gets
  closed in the xla emitter (see `docs/gap-analysis.md`); this repo only
  measures it. If you are about to add plumbing that ships the kernel,
  stop — that decision was made the other way on 2026-08-21.
- **Lazy-range invariants** in `benchmarks/handwritten/ntt.cu`: forward CT
  holds `[0, 4q)`, inverse GS `[0, 2q)`, one full reduction at the final
  store; both need `q < 2^60`. The all-`q−1` golden row exists to catch a
  broken invariant — keep it.
- **The split algebra is pinned in `forch/testing/split_test.py`** (pure
  Python, exact ints). Change the kernel's twiddle indexing only together
  with that test, never against it.
- **Trace boundary:** `Ring.poly`/`Poly.coeffs` are host-only (exact-int
  embedding, by lattice-frx design). Inside `frx.jit`, pass `Coeff`/`Eval`
  pytrees and build `Poly(ring, coeff=...)`. Domain policy: disjoint-domain
  add meets in Eval (see `trace_test.py` for the rationale).
- **The command-buffer mode is part of every number.** XLA's default (`LHS`)
  serializes independent fusions, so the 25-limb product reads 12.8 µs/NTT
  under it and 2.6 µs/NTT under
  `--xla_gpu_command_buffer_scheduling_mode=CONCURRENT` — a 5× swing.
  `benchmarks/ntt_bench.py` sets the flag itself (override with
  `FORCH_COMMAND_BUFFER_MODE`); never quote a number without saying which mode
  produced it. The flag costs ~2.2× peak memory (buffer assignment stops
  packing temporaries), which is why it is scoped here and not proposed as an
  XLA default — do not add it to unrelated repos' runs without measuring their
  peak memory first.
- **Limb grouping is the other switch that changes the numbers — but only the
  `Poly` product's.** An RNS ring element carries one dtype per limb, so no
  batch axis spans them; xla#569 lets the compiler merge the limbs' NTTs into
  one dispatch, and **xla#579 made that the backend default** (rationale and
  measurement: the 2026-08-22 update in `docs/gap-analysis.md`). So grouping is
  now on for anyone on an frx carrying that xla, `forch/` library paths
  included, with nobody asking. `benchmarks/ntt_bench.py` still sends the value
  per-computation (`FORCH_NTT_LIMB_GROUP`, default = the ring's limb count) to
  pin it, and prints one of three states — `on, max_group=N` / `off by request`
  / `unavailable (this frx predates xla#569)`. Quote that line with the
  product's µs/NTT. The sweep table above is single-limb `RnsRing([q], D)` jits
  with nothing to merge, so it is untouched by this switch; do not annotate
  those rows with it.
  - **An off switch must send the option, never just omit it** — omission means
    "whatever the wheel defaults to", not "off". See the comment in
    `compile_limb_grouped`, which this bit once. Applies to any
    upstream-defaulted knob this repo quotes a number against.
  - It is a **compile option, never `XLA_FLAGS`** — frxlib parses `XLA_FLAGS`
    against its own built-in list and `LOG(FATAL)`s on an unknown flag, so an
    older wheel would abort the run, where a compile option degrades to a
    catchable error the bench falls back from.
  - **Swapping the plugin `.so` tests a compile option, never a changed
    default.** frx ships frxlib's own `DebugOptions` whole, so a field that
    frxlib predates arrives unset and the plugin reads the proto default, not
    its own. A plugin swap therefore shows "no effect" for any default change
    and proves nothing — that needs a rebuilt frxlib, i.e. a wheel past the jax
    pin bump.
  - Why it can only live in the **benchmark**, not in `forch/`:
    `compiler_options` is rejected on a nested `jit` (`ValueError: can only be
    passed to top-level jax.jit`), and `Poly.__mul__` always runs inside the
    consumer's trace, so no library path can pass one. That is why the option
    is confined here — but it is no longer why the feature was unreachable:
    the default supplies it. Do not go looking for a peak-memory-style
    justification for the scoping; there isn't one.
- **Benchmarks:** warm, ≥30 reps, `block_until_ready` on a leaf
  (`.limbs[0]`), `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The GPU is shared on
  this machine — `bash benchmarks/run_all.sh` refuses to record while
  another process holds VRAM, and a "CUDA no device" error usually means
  VRAM is full, not a broken install. Golden files are generated
  (`make golden`), never committed.
- Tests are `absl.testing` (`absltest.main()` guard per file), same
  convention as lattice-frx.
