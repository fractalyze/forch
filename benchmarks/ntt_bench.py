#!/usr/bin/env python3
"""The opcode side of the table: lax.ntt raw / +bit_reverse / RnsRing.ntt,
and Poly end-to-end on an FGb-shaped 25-limb ring.

Run with the handwritten side:
    (cd benchmarks/handwritten && make check && make bench > ../results/hw.txt)
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python benchmarks/ntt_bench.py

Methodology: warm, 30 reps, block_until_ready on a leaf array. GPU may be
shared on this machine — check nvidia-smi before believing a slow number.
"""
import argparse
import os
import pathlib
import sys
import time

# XLA's default command-buffer mode (LHS) serializes independent fusions, which
# costs ~5x on an RNS workload: every limb's NTT waits for the previous one even
# though nothing connects them. CONCURRENT lets the CUDA graph's buffer-conflict
# DAG run them in parallel. Set before the backend initializes, and set here
# rather than documented, so a forgotten env var cannot silently publish a 5x
# regression as a measurement.
#
# The cost is real and is why this is not XLA's default: buffer assignment falls
# back to DependencyHloOrdering, whole-module heap simulation is skipped, and
# peak memory rises (measured 2.24x here, 51 -> 114 MiB). That is harmless at FHE
# sizes on a 32 GB card and would not be on a memory-bound prover, so the flag
# stays scoped to this benchmark instead of being pushed upstream as a default.
_MODE = os.environ.get("FORCH_COMMAND_BUFFER_MODE", "CONCURRENT")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + f" --xla_gpu_command_buffer_scheduling_mode={_MODE}"
).strip()

import numpy as np

import forch
import frx
import frx.numpy as fnp
import zk_dtypes
from frx import lax
from lattice_frx.primes import is_prime
from lattice_frx.ring import Coeff, RnsRing
from lattice_frx.roots import prime_factors, primitive_root

# The parameter set is owned by the golden generator so the handwritten and
# opcode halves of the table can never drift apart (both are benchmark code;
# test code stays non-API).
sys.path.insert(0, str(pathlib.Path(__file__).parent / "handwritten"))
from make_golden import CASES as PRIMES  # noqa: E402
from make_golden import D  # noqa: E402

BATCHES = (1, 16, 64, 256)


def ntt_primes_below(bits: int, count: int, d: int = D) -> list[int]:
    """Distinct NTT-friendly primes just under 2^bits (the OpenFHE LastPrime
    walk, continued). lattice-frx's own search caps at 50 bits, so the walk
    lives here for the 58-bit FGb shape."""
    out, q, step = [], (1 << bits) - (1 << bits) % (2 * d) + 1, 2 * d
    while len(out) < count:
        q -= step
        if is_prime(q):
            out.append(q)
    return out


def bench(fn, *args, reps: int) -> float:
    y = fn(*args)
    leaf = y.limbs[0] if hasattr(y, "limbs") else y
    leaf.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(reps):
        y = fn(*args)
    (y.limbs[0] if hasattr(y, "limbs") else y).block_until_ready()
    return (time.perf_counter() - t0) / reps


# Limb grouping (fractalyze/xla#569). An RNS ring element carries one
# prime_field(q_i) dtype per limb, so no tensor axis can span the limbs and each
# limb's transform is its own dispatch. Above 1, the compiler merges independent
# same-geometry NTT passes into one multi-root dispatch of k grid-z planes.
# Measured on the 25-limb product (RTX 5090, CONCURRENT, nsys GPU-busy): 150
# transform dispatches -> 6, NTT device time 215.4 -> 132.8 us, landing at 1.32x
# an equal-traffic single-modulus batched call. Byte-identical per limb, and the
# full attribution is in docs/gap-analysis.md.
#
# Asked for as a COMPILE OPTION, never through XLA_FLAGS: frxlib parses
# XLA_FLAGS against the flag list it was built with and calls LOG(FATAL) on an
# unknown one, so an frx older than the feature would abort every run in this
# file. As a compile option the same frx returns a catchable "No such compile
# option", so this runs on any wheel and switches itself on once frx carries an
# xla >= f7e9504.
_LIMB_GROUP_ENV = os.environ.get("FORCH_NTT_LIMB_GROUP")


def compile_limb_grouped(fn, *example_args, limbs: int):
    """`fn` compiled with limb grouping, and the state to quote alongside it.

    Returns `(compiled, state)`. `state` is the line the caller prints: the
    repo's rule is never to quote a number without the switch that produced it,
    and there are three states to tell apart, not two -- asked for and got it,
    asked for and this frx is too old, and deliberately turned off.

    The cap defaults to the ring's own limb count rather than a constant: that
    is the largest group this shape can ever form, so it never binds, and it
    stays honest when the CKKS layers push the limb count up.
    """
    group = int(_LIMB_GROUP_ENV) if _LIMB_GROUP_ENV else limbs
    # One lowering serves both branches, and the compile below is the only one
    # the run pays -- the caller times this executable directly. Passing the
    # option to BOTH jit() and compile() would duplicate the kv pair, miss the
    # executable cache, and silently compile the graph a second time.
    lowered = frx.jit(fn).lower(*example_args)
    if group < 2:
        return lowered.compile(), f"off by request (FORCH_NTT_LIMB_GROUP={group})"
    try:
        compiled = lowered.compile(
            compiler_options={"xla_gpu_ntt_max_fusion_group": group})
    except frx.errors.JaxRuntimeError as e:
        # Only the "frx predates the feature" case falls back. An OOM or a
        # compiler bug must not be recorded as a slower-but-valid number under
        # a line blaming the wheel's age.
        if "No such compile option" not in str(e):
            raise
        return lowered.compile(), "unavailable (this frx predates xla#569)"
    return compiled, f"on, max_group={group}"


def opcode_paths(q: int):
    field = zk_dtypes.prime_field(q)
    g = primitive_root(q, prime_factors(q - 1))
    ring1 = RnsRing([q], D)
    fwd = frx.jit(lambda x: lax.ntt(x, ntt_type=lax.NttType.NEGACYCLIC_NTT, generator=g))
    inv = frx.jit(lambda x: lax.ntt(x, ntt_type=lax.NttType.NEGACYCLIC_INTT, generator=g))
    brev = frx.jit(
        lambda x: lax.bit_reverse(
            lax.ntt(x, ntt_type=lax.NttType.NEGACYCLIC_NTT, generator=g), dimensions=(1,)
        )
    )
    # Wrapping the Coeff inside the jitted fn keeps every path's signature
    # uniform (a bare [batch, d] field array).
    ringntt = frx.jit(lambda x: ring1.ntt(Coeff((x,))))
    return field, {"lax.ntt raw fwd": fwd, "lax.ntt raw inv": inv,
                   "lax.ntt + bit_reverse": brev, "RnsRing.ntt (take)": ringntt}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=30)
    args = p.parse_args()
    reps = args.reps

    print(f"# frx {frx.__version__}, {frx.devices()}, d = 2^16, {reps} reps warm")
    print(f"# command-buffer scheduling mode: {_MODE}"
          f"  (XLA default is LHS; see the note at the top of this file)")
    requested = _LIMB_GROUP_ENV or "the ring's limb count"
    print(f"# limb grouping (Poly product only): {requested} requested")
    print("\n| path | prime | " + " | ".join(f"batch {b}" for b in BATCHES) + " |")
    print("|---" * (len(BATCHES) + 2) + "|")
    rng = np.random.default_rng(0)
    for name, q in PRIMES.items():
        field, paths = opcode_paths(q)
        rows = {k: [] for k in paths}
        for b in BATCHES:
            xh = rng.integers(0, q, (b, D), dtype=np.uint64)
            x = fnp.asarray(xh, dtype=field)
            for k, fn in paths.items():
                dt = bench(fn, x, reps=reps)
                rows[k].append(dt / b * 1e6)
        for k, vals in rows.items():
            print(f"| {k} | {name} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")

    # Poly end-to-end: FGb-shaped ring (1 x 58-bit + 24 x 42-bit limbs); the
    # limb axis is the batch. (a*b).coeffs-shaped graph, jitted, amortized
    # per transform (3 transforms per product).
    qs = ntt_primes_below(58, 1) + ntt_primes_below(42, 24)
    ring = forch.Ring(qs, D)
    ca = ring.poly(np.stack([rng.integers(0, q, D, dtype=np.uint64) for q in qs])).as_coeff()
    cb = ring.poly(np.stack([rng.integers(0, q, D, dtype=np.uint64) for q in qs])).as_coeff()

    # Returns the Coeff pytree rather than its .limbs so `bench` can pick the
    # leaf it blocks on, the same way the sweep's paths do.
    def product(a, b):
        return (forch.Poly(ring, coeff=a) * forch.Poly(ring, coeff=b)).as_coeff()

    f, grouping = compile_limb_grouped(product, ca, cb, limbs=len(qs))
    dt = bench(f, ca, cb, reps=reps)
    n_transforms = 3 * len(qs)
    peak = frx.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0)
    # The per-limb parenthetical stops being true once the limbs share a
    # dispatch, so say which one this run measured.
    dispatch = "one dispatch per pass" if grouping.startswith("on") else \
        "per-limb dispatch"
    print(f"\nPoly (a*b), {len(qs)}-limb FGb-shaped ring: {dt*1e3:.3f} ms/product, "
          f"{n_transforms} transforms -> {dt/n_transforms*1e6:.2f} us/NTT amortized "
          f"(includes pointwise mul + {dispatch})")
    print(f"limb grouping: {grouping}")
    # Process-wide peak, i.e. across the whole sweep above, not the product
    # alone — for the isolated 51 vs 114 MiB comparison see docs/gap-analysis.md.
    print(f"peak device memory this process: {peak/2**20:.1f} MiB "
          f"(mode {_MODE}; CONCURRENT trades ~2.2x peak for the overlap)")


if __name__ == "__main__":
    main()
