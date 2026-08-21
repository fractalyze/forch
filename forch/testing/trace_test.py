"""What compiles: transform counts and jit/eager parity.

The trace boundary is the `Coeff`/`Eval` container (a pytree of field
arrays), not the host array: `Ring.poly` embeds via exact Python integers,
which is a host-only operation by lattice-frx design. Inside a trace, build
`Poly(ring, coeff=...)` from containers passed as arguments.
"""
import numpy as np
from absl.testing import absltest

import forch
import frx

Q60 = 1152921504606584833
D = 64


class TraceTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.ring = forch.Ring([Q60], D)
        rng = np.random.default_rng(3)
        self.ca, self.cb, self.cc = (
            self.ring.poly(rng.integers(0, Q60, (1, D), dtype=np.uint64)).as_coeff()
            for _ in range(3)
        )

    def _p(self, c):
        return forch.Poly(self.ring, coeff=c)

    @staticmethod
    def _ntt_count(fn, *args) -> int:
        return str(frx.make_jaxpr(fn)(*args)).count("ntt[")

    def test_mul_is_three_transforms(self):
        def f(a, b):
            return (self._p(a) * self._p(b)).as_coeff().limbs

        self.assertEqual(self._ntt_count(f, self.ca, self.cb), 3)

    def test_add_after_mul_costs_at_most_one_embedding(self):
        # The product exists only in Eval; c only in Coeff. Meeting them costs
        # exactly one transform somewhere — forch meets in Eval (c is NTT'd),
        # the convention FHE schemes store ciphertexts in and the choice that
        # favors expressions that keep multiplying. The alternative (meet in
        # Coeff) would make THIS terminal expression 3 instead of 4 but pay
        # an extra transform on every (sum * e) continuation.
        def mul_only(a, b):
            return (self._p(a) * self._p(b)).as_coeff().limbs

        def mul_then_add(a, b, c):
            return ((self._p(a) * self._p(b)) + self._p(c)).as_coeff().limbs

        self.assertEqual(
            self._ntt_count(mul_then_add, self.ca, self.cb, self.cc),
            self._ntt_count(mul_only, self.ca, self.cb) + 1,
        )

    def test_reuse_transforms_once(self):
        def reuse(a, b):
            pa, pb = self._p(a), self._p(b)
            return ((pa * pb) + (pa * pb)).as_coeff().limbs  # pa/pb NTT'd once

        self.assertEqual(self._ntt_count(reuse, self.ca, self.cb), 2 + 1)

    def test_jit_matches_eager(self):
        def f(a, b, c):
            return ((self._p(a) * self._p(b)) + self._p(c)).as_coeff().limbs

        eager = f(self.ca, self.cb, self.cc)
        jitted = frx.jit(f)(self.ca, self.cb, self.cc)
        for e, j in zip(eager, jitted):
            np.testing.assert_array_equal(
                np.asarray(e).astype(np.uint64), np.asarray(j).astype(np.uint64)
            )


if __name__ == "__main__":
    absltest.main()
