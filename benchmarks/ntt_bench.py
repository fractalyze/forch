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
import pathlib
import sys
import time

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

    f = frx.jit(product)
    y = f(ca, cb); y[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(reps):
        y = f(ca, cb)
    y[0].block_until_ready()
    dt = (time.perf_counter() - t0) / reps
    n_transforms = 3 * len(qs)
    print(f"\nPoly (a*b), {len(qs)}-limb FGb-shaped ring: {dt*1e3:.3f} ms/product, "
          f"{n_transforms} transforms -> {dt/n_transforms*1e6:.2f} us/NTT amortized "
          f"(includes pointwise mul + per-limb dispatch)")


if __name__ == "__main__":
    main()
