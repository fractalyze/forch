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
- **Benchmarks:** warm, ≥30 reps, `block_until_ready` on a leaf
  (`.limbs[0]`), `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The GPU is shared on
  this machine — `bash benchmarks/run_all.sh` refuses to record while
  another process holds VRAM, and a "CUDA no device" error usually means
  VRAM is full, not a broken install. Golden files are generated
  (`make golden`), never committed.
- Tests are `absl.testing` (`absltest.main()` guard per file), same
  convention as lattice-frx.
