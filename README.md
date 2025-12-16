# gvsbench — GPU Vector Search Benchmark (CUDA)

Tested on A100 40GB VRAM with CUDA 13.0
Created with ChatGPT - GPT 5.2 Thinking

`gvsbench` is a simple command-line benchmark for **GPU vector search scoring** on Linux (Ubuntu 22.04 / 24.04). It measures throughput for a **brute-force inner-product (dot product) top-1 retrieval** workload, implemented as:

- **Chunked similarity scoring** via **cuBLAS SGEMM** (matrix multiplication)
- **Per-query reduction** on GPU to keep the **best match (top-1)**

This approximates the “dense retrieval scoring / reranking” primitive commonly used in vector search and retrieval systems.

---

## Features

- Pure CLI / CUI output suitable for SSH and servers
- Uses **cuBLAS** for high-performance GEMM
- Supports:
  - Resident DB in VRAM (when memory allows)
  - Streaming DB chunks (when DB is too large)
  - Optional **TF32** cuBLAS math mode on Ampere+ GPUs

---

## Requirements

- Ubuntu 22.04 or 24.04
- NVIDIA driver installed
- CUDA Toolkit installed (includes `nvcc` and cuBLAS)

Quick checks:

```bash
nvidia-smi
nvcc --version
```
```bash
Build:
nvcc -O3 -std=c++17 gvsbench.cu -lcublas -o gvsbench
```
```bash
Help:
./gvsbench --help
```
```bash
Typical Run:
./gvsbench --db 500000 --queries 1024 --dim 256 --chunk 16384 --runs 10 --warmup 2
```

