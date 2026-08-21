NVCC ?= /usr/local/cuda-12.9/bin/nvcc
ARCH ?= -arch=sm_120
NVCCFLAGS = -O3 -std=c++17 $(ARCH) -Xptxas -v

ntt_bench: ntt.cu harness.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

golden: make_golden.py
	../../.venv/bin/python make_golden.py

check: ntt_bench
	./ntt_bench golden/q60 check && ./ntt_bench golden/q50 check && \
	./ntt_bench golden/q58 check && ./ntt_bench golden/q42 check

bench: ntt_bench
	for c in q60 q50 q58 q42; do ./ntt_bench golden/$$c bench; done

.PHONY: golden check bench
