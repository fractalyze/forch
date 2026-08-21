"""The 2^8 x 2^8 split is twist-free with the psi^brev table.

This is the algebra the handwritten CUDA kernel implements; pinning it here
keeps kernel debugging about memory, not math. Pure CPU, exact ints.

Forward (CT-DIT, natural in -> bit-reversed out):
  phase 1: columns j (elements {j + 256k}), twiddle T[m + i]
  phase 2: chunks  c (elements [256c, 256c+256)), twiddle T[m*(256+c) + i]
Inverse (GS-DIF, bit-reversed in -> natural out): mirror order,
  phase 1: chunks with Tinv[m*(256+c) + i], phase 2: columns with
  Tinv[m + i], then every output scaled by d^{-1}.
"""
import numpy as np
from absl.testing import absltest

from lattice_frx.host_ring import HostRnsRing
from lattice_frx.roots import bit_reverse, prime_factors, primitive_root

Q = 1152921504606584833
N, LOGN, R = 1 << 16, 16, 256


def tables(q):
    g = primitive_root(q, prime_factors(q - 1))
    psi = pow(g, (q - 1) // (2 * N), q)
    psi_inv = pow(psi, q - 2, q)
    t, tinv = [0] * N, [0] * N
    for i in range(N):
        t[bit_reverse(i, LOGN)] = pow(psi, i, q)
        tinv[bit_reverse(i, LOGN)] = pow(psi_inv, i, q)
    return t, tinv


def ct256(a, tw, q):
    m, t = 1, R
    while m < R:
        t //= 2
        for i in range(m):
            w = tw(m, i)
            for j in range(i * 2 * t, i * 2 * t + t):
                u, v = a[j], a[j + t] * w % q
                a[j], a[j + t] = (u + v) % q, (u - v) % q
        m *= 2
    return a


def gs256(a, tw, q):
    m, t = R // 2, 1
    while m >= 1:
        for i in range(m):
            w = tw(m, i)
            for j in range(i * 2 * t, i * 2 * t + t):
                u, v = a[j], a[j + t]
                a[j], a[j + t] = (u + v) % q, (u - v) * w % q
        m //= 2
        t *= 2
    return a


class SplitTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.host = HostRnsRing([Q], N)
        self.t, self.tinv = tables(Q)
        rng = np.random.default_rng(0)
        self.x = rng.integers(0, Q, N, dtype=np.uint64)

    def test_forward_split(self):
        y = [int(v) for v in self.x]
        for j in range(R):
            col = ct256([y[j + R * k] for k in range(R)], lambda m, i: self.t[m + i], Q)
            for k in range(R):
                y[j + R * k] = col[k]
        for c in range(R):
            y[R * c:R * c + R] = ct256(
                y[R * c:R * c + R], lambda m, i, c=c: self.t[m * (R + c) + i], Q
            )
        want = self.host.ntt(self.x[None, :])[0]
        self.assertEqual(y, [int(v) for v in want])

    def test_inverse_split_roundtrips(self):
        fwd = self.host.ntt(self.x[None, :])[0]
        y = [int(v) for v in fwd]
        for c in range(R):
            y[R * c:R * c + R] = gs256(
                y[R * c:R * c + R], lambda m, i, c=c: self.tinv[m * (R + c) + i], Q
            )
        for j in range(R):
            col = gs256([y[j + R * k] for k in range(R)], lambda m, i: self.tinv[m + i], Q)
            for k in range(R):
                y[j + R * k] = col[k]
        n_inv = pow(N, Q - 2, Q)
        y = [v * n_inv % Q for v in y]
        want = self.host.intt(fwd[None, :])[0]
        self.assertEqual(y, [int(v) for v in want])
        self.assertEqual(y, [int(v) for v in self.x])


if __name__ == "__main__":
    absltest.main()
