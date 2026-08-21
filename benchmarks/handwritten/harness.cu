// harness — loads a golden case dir, checks the kernels byte-exact against
// the lattice-frx host ring, and times them with CUDA events.
//   ./ntt_bench <case_dir> check
//   ./ntt_bench <case_dir> bench
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using u64 = unsigned long long;

struct Tables { const u64 *t, *t_shoup; u64 q, n_inv, n_inv_shoup; };
extern "C" void run_forward(u64* d_data, const Tables* tb, int batch, cudaStream_t s);
extern "C" void run_inverse(u64* d_data, const Tables* tb, int batch, cudaStream_t s);

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(2); } } while (0)

static std::vector<u64> load(const std::string& path, size_t count) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "missing %s\n", path.c_str()); exit(2); }
  std::vector<u64> v(count);
  if (fread(v.data(), sizeof(u64), count, f) != count) { fprintf(stderr, "short read %s\n", path.c_str()); exit(2); }
  fclose(f);
  return v;
}

static u64 meta(const std::string& dir, const char* key) {
  FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
  if (!f) { fprintf(stderr, "missing meta.txt\n"); exit(2); }
  char line[128]; u64 val = 0; bool found = false;
  while (fgets(line, sizeof line, f)) {
    char* eq = strchr(line, '=');
    if (!eq) continue;
    *eq = 0;
    if (!strcmp(line, key)) { val = strtoull(eq + 1, nullptr, 10); found = true; }
  }
  fclose(f);
  if (!found) { fprintf(stderr, "meta key %s missing\n", key); exit(2); }
  return val;
}

static u64* to_dev(const std::vector<u64>& h) {
  u64* d;
  CUDA_OK(cudaMalloc(&d, h.size() * sizeof(u64)));
  CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(u64), cudaMemcpyHostToDevice));
  return d;
}

static int compare(const char* what, const std::vector<u64>& got, const std::vector<u64>& want, u64 q) {
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      fprintf(stderr, "FAIL %s at %zu: got %llu want %llu\n", what, i, got[i], want[i]);
      return 1;
    }
    if (got[i] >= q) { fprintf(stderr, "FAIL %s at %zu: %llu >= q\n", what, i, got[i]); return 1; }
  }
  printf("OK %s\n", what);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <case_dir> check|bench\n", argv[0]); return 2; }
  const std::string dir = argv[1], mode = argv[2];
  const u64 q = meta(dir, "q");
  const int d = (int)meta(dir, "d"), batch = (int)meta(dir, "batch");
  const size_t n = (size_t)batch * d;
  auto input = load(dir + "/input.bin", n), fwd = load(dir + "/fwd.bin", n);
  auto t = load(dir + "/t.bin", d), ts = load(dir + "/t_shoup.bin", d);
  auto ti = load(dir + "/tinv.bin", d), tis = load(dir + "/tinv_shoup.bin", d);

  Tables tb_f{to_dev(t), to_dev(ts), q, meta(dir, "n_inv"), meta(dir, "n_inv_shoup")};
  Tables tb_i{to_dev(ti), to_dev(tis), q, tb_f.n_inv, tb_f.n_inv_shoup};
  u64* data;
  CUDA_OK(cudaMalloc(&data, n * sizeof(u64)));
  std::vector<u64> out(n);
  int rc = 0;

  if (mode == "check") {
    CUDA_OK(cudaMemcpy(data, input.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
    run_forward(data, &tb_f, batch, 0);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(out.data(), data, n * sizeof(u64), cudaMemcpyDeviceToHost));
    rc |= compare((dir + " fwd").c_str(), out, fwd, q);

    CUDA_OK(cudaMemcpy(data, fwd.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
    run_inverse(data, &tb_i, batch, 0);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(out.data(), data, n * sizeof(u64), cudaMemcpyDeviceToHost));
    rc |= compare((dir + " inv").c_str(), out, input, q);

    CUDA_OK(cudaMemcpy(data, input.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
    run_forward(data, &tb_f, batch, 0);
    run_inverse(data, &tb_i, batch, 0);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(out.data(), data, n * sizeof(u64), cudaMemcpyDeviceToHost));
    rc |= compare((dir + " roundtrip").c_str(), out, input, q);
  } else {  // bench: batch sweep, tiling the golden input
    cudaEvent_t e0, e1;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));
    const int sweep[] = {1, 16, 64, 256};
    for (int b : sweep) {
      const size_t bn = (size_t)b * d;
      u64* big;
      if (cudaMalloc(&big, bn * sizeof(u64)) != cudaSuccess) {
        printf("skip batch %d (VRAM)\n", b);
        cudaGetLastError();
        continue;
      }
      for (int i = 0; i < b; ++i)
        CUDA_OK(cudaMemcpy(big + (size_t)i * d, input.data() + (size_t)(i % batch) * d,
                           d * sizeof(u64), cudaMemcpyHostToDevice));
      for (int dir_i = 0; dir_i < 2; ++dir_i) {
        auto run = dir_i ? run_inverse : run_forward;
        const Tables* tb = dir_i ? &tb_i : &tb_f;
        for (int w = 0; w < 5; ++w) run(big, tb, b, 0);
        CUDA_OK(cudaDeviceSynchronize());
        const int reps = 30;
        CUDA_OK(cudaEventRecord(e0));
        for (int r = 0; r < reps; ++r) run(big, tb, b, 0);
        CUDA_OK(cudaEventRecord(e1));
        CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
        double us_call = ms * 1000.0 / reps;
        printf("%s %s batch %3d  %9.2f us/call  %7.3f us/NTT\n", dir.c_str(),
               dir_i ? "inv" : "fwd", b, us_call, us_call / b);
      }
      CUDA_OK(cudaFree(big));
    }
    CUDA_OK(cudaEventDestroy(e0));
    CUDA_OK(cudaEventDestroy(e1));
  }
  return rc;
}
