"""Poly: multiplication and addition insert their own transforms.

Oracle is HostRnsRing: exact integers, lattigo order. The naive O(d^2)
negacyclic product at small d keeps the oracle itself honest.
"""
import numpy as np
from absl.testing import absltest, parameterized

import forch
from lattice_frx.host_ring import HostRnsRing

# 60-bit and 50-bit are the OpenFHE-default shapes; 58/42 (the HEaaN FGb
# shape) join in the GPU benchmark. Small d keeps the host oracle fast; the
# benchmark is where d = 2^16 lives. Both primes are ≡ 1 mod 2*64.
Q60 = 1152921504606584833
Q50 = 1125899908022273
D = 64


def naive_negacyclic(a, b, q, d):
    out = [0] * d
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            k = i + j
            v = ai * bj
            if k >= d:
                out[k - d] = (out[k - d] - v) % q
            else:
                out[k] = (out[k] + v) % q
    return out


class PolyTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.qs = (Q60, Q50)
        self.ring = forch.Ring(self.qs, D)
        self.host = HostRnsRing(self.qs, D)
        self.rng = np.random.default_rng(7)

    def rand(self):
        return np.stack([self.rng.integers(0, q, D, dtype=np.uint64) for q in self.qs])

    def test_mul_matches_host_ntt_product(self):
        a, b = self.rand(), self.rand()
        got = (self.ring.poly(a) * self.ring.poly(b)).coeffs()
        want = self.host.intt(self.host.mul(self.host.ntt(a), self.host.ntt(b)))
        np.testing.assert_array_equal(got, want)

    def test_mul_matches_naive_negacyclic(self):
        a, b = self.rand(), self.rand()
        got = (self.ring.poly(a) * self.ring.poly(b)).coeffs()
        for li, q in enumerate(self.qs):
            self.assertEqual(
                list(map(int, got[li])),
                naive_negacyclic([int(x) for x in a[li]], [int(x) for x in b[li]], q, D),
            )

    def test_add_sub_in_coeff_domain(self):
        a, b = self.rand(), self.rand()
        np.testing.assert_array_equal(
            (self.ring.poly(a) + self.ring.poly(b)).coeffs(), self.host.add(a, b)
        )
        np.testing.assert_array_equal(
            (self.ring.poly(a) - self.ring.poly(b)).coeffs(), self.host.sub(a, b)
        )

    def test_mixed_domain_add_after_mul(self):
        a, b, c = self.rand(), self.rand(), self.rand()
        got = (self.ring.poly(a) * self.ring.poly(b) + self.ring.poly(c)).coeffs()
        want = self.host.add(
            self.host.intt(self.host.mul(self.host.ntt(a), self.host.ntt(b))), c
        )
        np.testing.assert_array_equal(got, want)

    def test_mul_add(self):
        a, b, c, e = (self.rand() for _ in range(4))
        acc = self.ring.poly(c) * self.ring.poly(e)
        got = self.ring.poly(a).mul_add(self.ring.poly(b), acc).coeffs()
        want = self.host.intt(
            self.host.mul_add(
                self.host.ntt(a),
                self.host.ntt(b),
                self.host.mul(self.host.ntt(c), self.host.ntt(e)),
            )
        )
        np.testing.assert_array_equal(got, want)

    def test_from_signed_and_explicit_transforms(self):
        vals = [3, -1, 0, 2] + [0] * (D - 4)
        p = self.ring.from_signed(vals)
        np.testing.assert_array_equal(p.ntt().intt().coeffs(), p.coeffs())

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            self.ring.poly(self.rand()[:, : D // 2])


if __name__ == "__main__":
    absltest.main()
