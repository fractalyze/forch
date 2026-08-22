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

# Limb grouping (fractalyze/xla#569). An RNS ring element carries one
# prime_field(q_i) dtype per limb, so no tensor axis can span the limbs and each
# limb's transform is its own dispatch. Above 1, the compiler merges independent
# same-geometry NTT passes into one multi-root dispatch of k grid-z planes.
# Measured on the 25-limb product below (RTX 5090, CONCURRENT, nsys GPU-busy):
# 150 transform dispatches -> 6, NTT device time 215.4 -> 132.8 us, landing at
# 1.32x an equal-traffic single-modulus batched call. Byte-identical per limb.
#
# Passed as a COMPILE OPTION, not through XLA_FLAGS: frxlib parses XLA_FLAGS
# against the flag list it was built with and calls LOG(FATAL) on an unknown
# one, so an frx wheel older than the feature would abort every run in this
# file. As a compile option the same wheel returns a catchable
# "No such compile option", which is what the fallback below reads -- so this
# runs on any wheel and switches itself on once frx carries an xla >= f7e9504.
_LIMB_GROUP = int(os.environ.get("FORCH_NTT_LIMB_GROUP", "64"))

import numpy as np

import forch
import frx
import frx.errors
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


def jit_limb_grouped(fn, *example_args):
    """`frx.jit(fn)` with limb grouping on, or plain jit when frx predates it.

    Returns `(compiled, grouped)`; `grouped` says which of the two produced the
    numbers, since the repo's rule is never to quote one without the mode.
    """
    if _LIMB_GROUP > 1:
        opts = {"xla_gpu_ntt_max_fusion_group": _LIMB_GROUP}
        try:
            compiled = frx.jit(fn, compiler_options=opts)
            # Force the compile here: jit is lazy, and an frx that does not know
            # the option only says so when it first compiles.
            compiled.lower(*example_args).compile(compiler_options=opts)
            return compiled, True
        except frx.errors.JaxRuntimeError:
            pass
    return frx.jit(fn), False


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

    def product(a, b):
        return (forch.Poly(ring, coeff=a) * forch.Poly(ring, coeff=b)).as_coeff().limbs

    f, grouped = jit_limb_grouped(product, ca, cb)
    y = f(ca, cb); y[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(reps):
        y = f(ca, cb)
    y[0].block_until_ready()
    dt = (time.perf_counter() - t0) / reps
    n_transforms = 3 * len(qs)
    peak = frx.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0)
    print(f"\nPoly (a*b), {len(qs)}-limb FGb-shaped ring: {dt*1e3:.3f} ms/product, "
          f"{n_transforms} transforms -> {dt/n_transforms*1e6:.2f} us/NTT amortized "
          f"(includes pointwise mul + per-limb dispatch)")
    print("limb grouping: "
          + (f"on, max_group={_LIMB_GROUP}" if grouped else
             "OFF -- this frx predates xla#569, so each limb is still its own "
             "dispatch; upgrade frx to collapse them"))
    # Process-wide peak, i.e. across the whole sweep above, not the product
    # alone — for the isolated 51 vs 114 MiB comparison see docs/gap-analysis.md.
    print(f"peak device memory this process: {peak/2**20:.1f} MiB "
          f"(mode {_MODE}; CONCURRENT trades ~2.2x peak for the overlap)")


if __name__ == "__main__":
    main()
