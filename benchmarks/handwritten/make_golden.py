#!/usr/bin/env python3
"""Dump golden NTT vectors + twiddle tables for the handwritten harness.

Everything derives from lattice-frx (generator walk, host ring), so the
binary's byte-exact gate is against the same oracle as the Python suite.
"""
import argparse
import pathlib

import numpy as np
from lattice_frx.host_ring import HostRnsRing
from lattice_frx.roots import bit_reverse, prime_factors, primitive_root

CASES = {  # OpenFHE-default (60/50) and HEaaN-FGb (58/42) shapes, all ≡ 1 mod 2^17
    "q60": 1152921504606584833,  # LastPrime(60, 2^17) = 2^60 - 2^18 + 1
    "q50": 1125899908022273,     # FirstPrime(50, 2^17)
    "q58": 288230376147386369,   # LastPrime(58, 2^17) = 0x3ffffffffbe0001
    "q42": 4398044938241,        # LastPrime(42, 2^17) = 0x3ffffe80001
}


def shoup(w: int, q: int) -> int:
    return (w << 64) // q


def dump(out: pathlib.Path, q: int, d: int, batch: int, seed: int) -> None:
    logn = d.bit_length() - 1
    g = primitive_root(q, prime_factors(q - 1))
    psi = pow(g, (q - 1) // (2 * d), q)
    psi_inv = pow(psi, q - 2, q)
    t, tinv = [0] * d, [0] * d
    for i in range(d):
        t[bit_reverse(i, logn)] = pow(psi, i, q)
        tinv[bit_reverse(i, logn)] = pow(psi_inv, i, q)
    host = HostRnsRing([q], d)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, q, (batch, d), dtype=np.uint64)
    # Adversarial rows for the lazy-range probe: all q-1, all zero.
    x[0, :] = q - 1
    if batch > 1:
        x[1, :] = 0
    fwd = np.stack([host.ntt(x[i : i + 1])[0] for i in range(batch)])

    out.mkdir(parents=True, exist_ok=True)
    n_inv = pow(d, q - 2, q)
    (out / "meta.txt").write_text(
        f"q={q}\nd={d}\nbatch={batch}\nn_inv={n_inv}\nn_inv_shoup={shoup(n_inv, q)}\n"
    )
    x.astype("<u8").tofile(out / "input.bin")
    fwd.astype("<u8").tofile(out / "fwd.bin")
    for name, tab in (("t", t), ("tinv", tinv)):
        np.array(tab, dtype="<u8").tofile(out / f"{name}.bin")
        np.array([shoup(int(w), q) for w in tab], dtype="<u8").tofile(
            out / f"{name}_shoup.bin"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(pathlib.Path(__file__).parent / "golden"))
    p.add_argument("--d", type=int, default=1 << 16)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()
    for name, q in CASES.items():
        assert (q - 1) % (2 * args.d) == 0, name
        dump(pathlib.Path(args.out) / name, q, args.d, args.batch, seed=42)
        print("wrote", name)
