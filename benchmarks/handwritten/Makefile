NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda-12.9/bin/nvcc)
PYTHON ?= ../../.venv/bin/python
ARCH ?= -arch=sm_120
NVCCFLAGS = -O3 -std=c++17 $(ARCH) -Xptxas -v
CASES = q60 q50 q58 q42

ntt_bench: ntt.cu harness.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

golden: make_golden.py
	$(PYTHON) make_golden.py

check: ntt_bench
	for c in $(CASES); do ./ntt_bench golden/$$c check || exit 1; done

bench: ntt_bench
	for c in $(CASES); do ./ntt_bench golden/$$c bench; done

.PHONY: golden check bench
