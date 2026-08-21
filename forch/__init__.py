"""forch — FHE evaluation on the lattice-frx substrate.

The one idea in v0: user code multiplies polynomials; the transforms are
inserted here, at trace time, and the whole expression compiles through the
`ntt` opcode. Domain bookkeeping is Python-side state on `Poly` — consistent
with lattice-frx's "the domain is a type": forch only chooses which
statically-typed op to emit, it never branches on a traced value.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from lattice_frx import ring as _ring

__version__ = "0.0.1"
__all__ = ["Ring", "Poly"]


class Ring:
    """`Z_q[X]/(X^d + 1)` over an RNS chain, with an auto-transforming Poly."""

    def __init__(self, q_moduli: Sequence[int], d: int) -> None:
        self._ring = _ring.RnsRing(q_moduli, d)
        self.q_moduli = self._ring.q_moduli
        self.d = d

    def poly(self, arr: np.ndarray) -> "Poly":
        """A `(limbs, d)` uint64 host array as a coefficient-domain Poly.

        The substrate's constructor takes the rows on faith; the shape check
        lives here because a truncated row would otherwise build a shorter
        ring element silently.
        """
        expected = (len(self.q_moduli), self.d)
        if np.shape(arr) != expected:
            raise ValueError(f"poly: expected shape {expected}, got {np.shape(arr)}")
        return Poly(self, coeff=self._ring.coeff_from_host(arr))

    def from_signed(self, values) -> "Poly":
        return Poly(self, coeff=self._ring.from_signed(values))


class Poly:
    """A ring element that caches both domains and converts lazily.

    `_coeff` / `_eval` hold the lattice-frx containers; at least one is set.
    Conversions cache: a Poly used twice transforms once per trace.
    """

    def __init__(self, ring: Ring, *, coeff=None, eval_=None) -> None:
        if coeff is None and eval_ is None:
            raise ValueError("Poly needs at least one domain")
        self.ring = ring
        self._coeff: Optional[_ring.Coeff] = coeff
        self._eval: Optional[_ring.Eval] = eval_

    # -- domain access (cached) ------------------------------------------
    def _as_coeff(self) -> _ring.Coeff:
        if self._coeff is None:
            self._coeff = self.ring._ring.intt(self._eval)
        return self._coeff

    def _as_eval(self) -> _ring.Eval:
        if self._eval is None:
            self._eval = self.ring._ring.ntt(self._coeff)
        return self._eval

    def ntt(self) -> "Poly":
        self._as_eval()
        return self

    def intt(self) -> "Poly":
        self._as_coeff()
        return self

    # -- arithmetic ------------------------------------------------------
    def __mul__(self, other: "Poly") -> "Poly":
        self._same_ring(other)
        return Poly(self.ring, eval_=self.ring._ring.mul(self._as_eval(), other._as_eval()))

    def mul_add(self, other: "Poly", acc: "Poly") -> "Poly":
        self._same_ring(other)
        self._same_ring(acc)
        return Poly(
            self.ring,
            eval_=self.ring._ring.mul_add(self._as_eval(), other._as_eval(), acc._as_eval()),
        )

    def _addsub(self, other: "Poly", op) -> "Poly":
        self._same_ring(other)
        # Meet in a domain both sides already have; a shared one costs no
        # transform. Otherwise converge on Eval (either choice inserts one
        # transform; Eval is the domain the next mul wants).
        if self._coeff is not None and other._coeff is not None:
            return Poly(self.ring, coeff=op(self._coeff, other._coeff))
        return Poly(self.ring, eval_=op(self._as_eval(), other._as_eval()))

    def __add__(self, other: "Poly") -> "Poly":
        return self._addsub(other, self.ring._ring.add)

    def __sub__(self, other: "Poly") -> "Poly":
        return self._addsub(other, self.ring._ring.sub)

    # -- host boundary ---------------------------------------------------
    def coeffs(self) -> np.ndarray:
        return self.ring._ring.to_host(self._as_coeff())

    def _same_ring(self, other: "Poly") -> None:
        if other.ring is not self.ring:
            raise ValueError("Polys belong to different rings")
