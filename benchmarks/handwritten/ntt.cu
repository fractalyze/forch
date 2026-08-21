// ntt.cu — negacyclic NTT at d = 2^16 = 2^8 x 2^8, the FHE-literature design:
//   * Shoup multiply + Harvey lazy ranges (Harvey JSC'14, arXiv 1205.2926;
//     Longa–Naehrig CANS'16, eprint 2016/504): forward CT keeps [0,4q),
//     inverse GS keeps [0,2q), one full reduction at the final store; q < 2^60.
//   * psi^brev twiddle tables (twist merged; Roy+ CHES'14, POG'15): forward is
//     natural -> bit-reversed (= the lattice-frx / lattigo contract order),
//     inverse takes it back; no permutation is ever materialized.
//   * Two phases, twist-free with this table (pinned by forch's split_test.py):
//     forward: columns {j+256k} with T[m + i], then chunks c with
//     T[m*(256+c) + i]; inverse mirrors in reverse order over Tinv, then a
//     final x d^{-1} folded into the last store (simpler than OpenFHE's
//     half-fold, invisible under the memory bound).
//   * One warp per 256-pt sub-transform, 8 elements per lane {lane + 32r}:
//     3 register stages (partner r^4, r^2, r^1) + 5 shuffle stages (partner
//     lane^16..lane^1). No __syncthreads inside a transform.
#include <cuda_runtime.h>

using u64 = unsigned long long;

struct Tables { const u64 *t, *t_shoup; u64 q, n_inv, n_inv_shoup; };

__device__ __forceinline__ u64 mul_shoup(u64 x, u64 w, u64 ws, u64 q) {
  return x * w - __umul64hi(x, ws) * q;  // any u64 x, w < q  ->  [0, 2q)
}

// ---------------------------------------------------------------------------
// Forward: 256-pt CT-DIT in a warp. Twiddle for stage m, group i: t[m*base+i].
// Values in [0, 4q) throughout.
__device__ __forceinline__ void ntt256_warp(u64 v[8], const Tables& tb, u64 base) {
  const u64 q = tb.q, two_q = 2 * tb.q;
  const unsigned lane = threadIdx.x & 31u;
  // Register stages: m = 1, 2, 4; t = 128, 64, 32; partner r ^ (4 >> s).
  for (unsigned s = 0; s < 3; ++s) {
    const unsigned m = 1u << s, t = 128u >> s, rmask = 4u >> s;
    for (unsigned r = 0; r < 8; ++r) {
      if (r & rmask) continue;  // r holds x, r|rmask holds y
      const unsigned i = (lane + 32u * r) / (2u * t);
      const u64 idx = (u64)m * base + i;
      const u64 w = __ldg(&tb.t[idx]), ws = __ldg(&tb.t_shoup[idx]);
      u64 &x = v[r], &y = v[r | rmask];
      if (x >= two_q) x -= two_q;
      const u64 tv = mul_shoup(y, w, ws, q);
      y = x + two_q - tv;
      x = x + tv;
    }
  }
  // Shuffle stages: m = 8..128; t = 16..1; partner lane ^ t.
  for (unsigned s = 3; s < 8; ++s) {
    const unsigned m = 1u << s, t = 128u >> s;
    for (unsigned r = 0; r < 8; ++r) {
      const unsigned i = (lane + 32u * r) / (2u * t);
      const u64 idx = (u64)m * base + i;
      const bool hi = (lane & t) != 0;  // hi lane holds y
      u64 send;
      if (hi) {
        send = mul_shoup(v[r], __ldg(&tb.t[idx]), __ldg(&tb.t_shoup[idx]), q);
      } else {
        if (v[r] >= two_q) v[r] -= two_q;
        send = v[r];
      }
      const u64 got = __shfl_xor_sync(0xffffffffu, send, t);
      // hi: got = partner's reduced x, send = own t-value -> x - t + 2q.
      // lo: got = partner's t-value                       -> x + t.
      v[r] = hi ? got + two_q - send : v[r] + got;
    }
  }
}

// ---------------------------------------------------------------------------
// Inverse: 256-pt GS-DIF in a warp, shuffle stages first (m = 128 -> 8),
// then register stages (m = 4, 2, 1). Values in [0, 2q) throughout.
__device__ __forceinline__ void intt256_warp(u64 v[8], const Tables& tb, u64 base) {
  const u64 q = tb.q, two_q = 2 * tb.q;
  const unsigned lane = threadIdx.x & 31u;
  for (unsigned s = 7; s >= 3; --s) {  // t = 1..16: partner lane ^ t
    const unsigned m = 1u << s, t = 128u >> s;
    for (unsigned r = 0; r < 8; ++r) {
      const unsigned i = (lane + 32u * r) / (2u * t);
      const u64 idx = (u64)m * base + i;
      const bool hi = (lane & t) != 0;  // hi lane holds y
      const u64 got = __shfl_xor_sync(0xffffffffu, v[r], t);
      if (hi) {  // y' = (x - y + 2q) * w, x arrives via shuffle
        v[r] = mul_shoup(got + two_q - v[r], __ldg(&tb.t[idx]), __ldg(&tb.t_shoup[idx]), q);
      } else {   // x' = x + y, reduced back to [0, 2q)
        u64 u = v[r] + got;
        if (u >= two_q) u -= two_q;
        v[r] = u;
      }
    }
  }
  for (int s = 2; s >= 0; --s) {  // t = 32, 64, 128: partner r ^ (4 >> s)
    const unsigned m = 1u << s, t = 128u >> s, rmask = 4u >> s;
    for (unsigned r = 0; r < 8; ++r) {
      if (r & rmask) continue;
      const unsigned i = (lane + 32u * r) / (2u * t);
      const u64 idx = (u64)m * base + i;
      u64 &x = v[r], &y = v[r | rmask];
      u64 u = x + y;
      if (u >= two_q) u -= two_q;
      y = mul_shoup(x + two_q - y, __ldg(&tb.t[idx]), __ldg(&tb.t_shoup[idx]), q);
      x = u;
    }
  }
}

__device__ __forceinline__ u64 reduce4q(u64 x, u64 q) {
  if (x >= 2 * q) x -= 2 * q;
  if (x >= q) x -= q;
  return x;
}

#ifndef T_COLS
#define T_COLS 16
#endif
#ifndef W_CHUNKS
#define W_CHUNKS 8
#endif

// ---------------------------------------------------------------------------
// Column kernels: a block stages a [256][T_COLS] strided tile through padded
// shared memory (coalesced global rows; +1 pad halves 64-bit bank conflicts),
// one warp per column. Grid: (256/T_COLS, batch).
template <bool kForward>
__global__ void column_kernel(u64* data, Tables tb) {
  __shared__ u64 sh[256][T_COLS + 1];
  u64* poly = data + (u64)blockIdx.y * 65536u;
  const unsigned j0 = blockIdx.x * T_COLS, nthr = T_COLS * 32u;
  for (unsigned e = threadIdx.x; e < 256u * T_COLS; e += nthr)
    sh[e / T_COLS][e % T_COLS] = poly[(u64)(e / T_COLS) * 256u + j0 + (e % T_COLS)];
  __syncthreads();
  const unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  u64 v[8];
  for (unsigned r = 0; r < 8; ++r) v[r] = sh[lane + 32u * r][w];
  if (kForward) {
    ntt256_warp(v, tb, 1u);
  } else {
    intt256_warp(v, tb, 1u);
    // Last inverse phase: fold in d^{-1} and reduce to canonical [0, q).
    for (unsigned r = 0; r < 8; ++r) {
      u64 x = mul_shoup(v[r], tb.n_inv, tb.n_inv_shoup, tb.q);
      v[r] = x >= tb.q ? x - tb.q : x;
    }
  }
  for (unsigned r = 0; r < 8; ++r) sh[lane + 32u * r][w] = v[r];
  __syncthreads();
  for (unsigned e = threadIdx.x; e < 256u * T_COLS; e += nthr)
    poly[(u64)(e / T_COLS) * 256u + j0 + (e % T_COLS)] = sh[e / T_COLS][e % T_COLS];
}

// Chunk kernels: warp w owns contiguous chunk c = blockIdx.x*W_CHUNKS + w;
// fully coalesced direct loads, no shared memory, no block barrier.
// Grid: (256/W_CHUNKS, batch).
template <bool kForward>
__global__ void chunk_kernel(u64* data, Tables tb) {
  u64* poly = data + (u64)blockIdx.y * 65536u;
  const unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  const unsigned c = blockIdx.x * W_CHUNKS + w;
  u64* chunk = poly + (u64)c * 256u;
  u64 v[8];
  for (unsigned r = 0; r < 8; ++r) v[r] = chunk[lane + 32u * r];
  if (kForward) {
    ntt256_warp(v, tb, 256u + c);
    // Final forward phase: reduce to canonical [0, q).
    for (unsigned r = 0; r < 8; ++r) v[r] = reduce4q(v[r], tb.q);
  } else {
    intt256_warp(v, tb, 256u + c);  // first inverse phase: stay lazy [0, 2q)
  }
  for (unsigned r = 0; r < 8; ++r) chunk[lane + 32u * r] = v[r];
}

extern "C" void run_forward(u64* d_data, const Tables* tb, int batch, cudaStream_t s) {
  dim3 g1(256 / T_COLS, batch), g2(256 / W_CHUNKS, batch);
  column_kernel<true><<<g1, T_COLS * 32, 0, s>>>(d_data, *tb);
  chunk_kernel<true><<<g2, W_CHUNKS * 32, 0, s>>>(d_data, *tb);
}

extern "C" void run_inverse(u64* d_data, const Tables* tb, int batch, cudaStream_t s) {
  dim3 g1(256 / W_CHUNKS, batch), g2(256 / T_COLS, batch);
  chunk_kernel<false><<<g1, W_CHUNKS * 32, 0, s>>>(d_data, *tb);
  column_kernel<false><<<g2, T_COLS * 32, 0, s>>>(d_data, *tb);
}
