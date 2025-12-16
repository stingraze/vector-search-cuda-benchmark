// gvsbench.cu
// GPU Vector Search Benchmark (Inner Product, Top-1) using cuBLAS SGEMM + CUDA reduction.
// Designed to compile cleanly with nvcc on Ubuntu 22.04/24.04.
// (C)Tsubasa Kato - Created using ChatGPT - GPT 5.2 Thinking
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <getopt.h>

#define CUDA_CHECK(call) do {                                  \
  cudaError_t _e = (call);                                     \
  if (_e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(_e));       \
    std::exit(1);                                              \
  }                                                           \
} while(0)

#define CUBLAS_CHECK(call) do {                                \
  cublasStatus_t _s = (call);                                  \
  if (_s != CUBLAS_STATUS_SUCCESS) {                           \
    fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",         \
            __FILE__, __LINE__, int(_s));                      \
    std::exit(1);                                              \
  }                                                           \
} while(0)

// Hash available to device code (fixes your compile error)
static __host__ __device__ __forceinline__
uint32_t wang_hash(uint32_t x) {
  x = (x ^ 61u) ^ (x >> 16);
  x *= 9u;
  x = x ^ (x >> 4);
  x *= 0x27d4eb2du;
  x = x ^ (x >> 15);
  return x;
}

// Fill a flat array with deterministic pseudo-random floats in [-1, 1].
// Grid-stride loop avoids huge grid dims and index overflow.
__global__ void fill_random_flat(float* A, size_t n, uint32_t seed) {
  size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(blockDim.x) * gridDim.x;

  for (size_t i = tid; i < n; i += stride) {
    uint32_t h = wang_hash(uint32_t(i) ^ seed);
    float u = (h & 0x00FFFFFFu) / float(0x01000000u);  // [0,1)
    A[i] = 2.0f * u - 1.0f;                             // [-1,1)
  }
}

__global__ void init_best(float* best_scores, int* best_idx, int nq) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q < nq) {
    best_scores[q] = -INFINITY;
    best_idx[q] = -1;
  }
}

// Reduce row-wise max for S of shape (nq x chunk), column-major (ld = nq).
// Updates global best arrays with db_offset.
__global__ void reduce_row_max_update(
    const float* __restrict__ S,
    int nq, int chunk,
    float* __restrict__ best_scores,
    int* __restrict__ best_idx,
    int db_offset)
{
  // One block per query row
  int q = blockIdx.x;
  if (q >= nq) return;

  // This kernel is written for 256 threads per block.
  // If you change the launch block size, change the shared sizes accordingly.
  float local_max = -INFINITY;
  int local_arg = -1;

  for (int j = threadIdx.x; j < chunk; j += blockDim.x) {
    float v = S[q + nq * j];
    if (v > local_max) { local_max = v; local_arg = j; }
  }

  __shared__ float smax[256];
  __shared__ int   sarg[256];

  int t = threadIdx.x;
  smax[t] = local_max;
  sarg[t] = local_arg;
  __syncthreads();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (t < stride) {
      float v2 = smax[t + stride];
      int   a2 = sarg[t + stride];
      if (v2 > smax[t]) { smax[t] = v2; sarg[t] = a2; }
    }
    __syncthreads();
  }

  if (t == 0) {
    float prev = best_scores[q];
    float cand = smax[0];
    if (cand > prev) {
      best_scores[q] = cand;
      best_idx[q] = db_offset + sarg[0];
    }
  }
}

static void print_line(int w=78) {
  for (int i=0;i<w;i++) std::printf("-");
  std::printf("\n");
}

static std::string bytes_human(double b) {
  const char* suf[] = {"B","KB","MB","GB","TB"};
  int i=0;
  while (b >= 1024.0 && i < 4) { b/=1024.0; i++; }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f %s", b, suf[i]);
  return std::string(buf);
}

static void usage() {
  std::printf(
R"(gvsbench - GPU Vector Search Benchmark (CUDA/cuBLAS)

Usage:
  ./gvsbench [options]

Options:
  --device N         GPU device index (default: 0)
  --db N             number of database vectors (default: 262144)
  --queries N         number of query vectors (default: 1024)
  --dim N            vector dimension (default: 256)
  --chunk N          database chunk size for GEMM (default: 16384)
  --runs N           timed runs (default: 10)
  --warmup N         warmup runs (default: 2)
  --tf32 0|1         enable TF32 Tensor Core math in cuBLAS (default: 0)
  --resident -1|0|1  -1 auto (default), 0 force streaming, 1 force resident DB in VRAM
  --help             show this help

What it does:
  Computes top-1 inner product matches:
    best[q] = max_i dot(Q[q], X[i])
  using chunked SGEMM + per-query reduction.
)");
}

int main(int argc, char** argv) {
  int device = 0;
  long long db = 262144;
  int nq = 1024;
  int dim = 256;
  int chunk = 16384;
  int runs = 10;
  int warmup = 2;
  int tf32 = 0;
  int resident_flag = -1; // -1 auto, 0 force stream, 1 force resident

  static option long_opts[] = {
    {"device",   required_argument, nullptr, 0},
    {"db",       required_argument, nullptr, 0},
    {"queries",  required_argument, nullptr, 0},
    {"dim",      required_argument, nullptr, 0},
    {"chunk",    required_argument, nullptr, 0},
    {"runs",     required_argument, nullptr, 0},
    {"warmup",   required_argument, nullptr, 0},
    {"tf32",     required_argument, nullptr, 0},
    {"resident", required_argument, nullptr, 0},
    {"help",     no_argument,       nullptr, 0},
    {nullptr, 0, nullptr, 0}
  };

  while (true) {
    int idx = 0;
    int c = getopt_long(argc, argv, "", long_opts, &idx);
    if (c == -1) break;
    if (c != 0) continue;

    std::string opt = long_opts[idx].name;
    if (opt == "device") device = std::atoi(optarg);
    else if (opt == "db") db = std::atoll(optarg);
    else if (opt == "queries") nq = std::atoi(optarg);
    else if (opt == "dim") dim = std::atoi(optarg);
    else if (opt == "chunk") chunk = std::atoi(optarg);
    else if (opt == "runs") runs = std::atoi(optarg);
    else if (opt == "warmup") warmup = std::atoi(optarg);
    else if (opt == "tf32") tf32 = std::atoi(optarg);
    else if (opt == "resident") resident_flag = std::atoi(optarg);
    else if (opt == "help") { usage(); return 0; }
  }

  if (db <= 0 || nq <= 0 || dim <= 0 || chunk <= 0 || runs <= 0 || warmup < 0) {
    std::fprintf(stderr, "Invalid arguments. Use --help.\n");
    return 1;
  }
  if (chunk > db) chunk = (int)db;

  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  size_t freeB=0, totalB=0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

  // Memory estimates
  size_t bytesQ      = size_t(nq) * size_t(dim) * sizeof(float);                 // Q (nq x dim)
  size_t bytesS      = size_t(nq) * size_t(chunk) * sizeof(float);               // S (nq x chunk)
  size_t bytesBest   = size_t(nq) * (sizeof(float) + sizeof(int));               // best score + idx
  size_t bytesDBFull = size_t(dim) * size_t(db) * sizeof(float);                 // DB^T (dim x db)

  bool resident = false;
  if (resident_flag == 1) resident = true;
  else if (resident_flag == 0) resident = false;
  else {
    double need = double(bytesQ + bytesS + bytesBest + bytesDBFull);
    resident = (need < double(freeB) * 0.70);
  }
  if (resident && (bytesQ + bytesS + bytesBest + bytesDBFull > freeB)) {
    resident = false;
  }

  float* dQ = nullptr;
  float* dS = nullptr;
  float* dBest = nullptr;
  int*   dBestIdx = nullptr;

  CUDA_CHECK(cudaMalloc(&dQ, bytesQ));
  CUDA_CHECK(cudaMalloc(&dS, bytesS));
  CUDA_CHECK(cudaMalloc(&dBest, nq * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dBestIdx, nq * sizeof(int)));

  float* dDBT_full = nullptr;
  float* dDBT_chunk = nullptr;

  if (resident) CUDA_CHECK(cudaMalloc(&dDBT_full, bytesDBFull));
  else CUDA_CHECK(cudaMalloc(&dDBT_chunk, size_t(dim) * size_t(chunk) * sizeof(float)));

  // Reasonable grid sizing for fill kernel
  int fill_threads = 256;
  int fill_blocks = std::min( (prop.multiProcessorCount * 8), 65535 );

  // Fill Q
  CUDA_CHECK(cudaMemset(dQ, 0, bytesQ));
  fill_random_flat<<<fill_blocks, fill_threads>>>(dQ, size_t(nq) * size_t(dim), 12345u);
  CUDA_CHECK(cudaGetLastError());

  // If resident, fill full DB^T once
  if (resident) {
    fill_random_flat<<<fill_blocks, fill_threads>>>(dDBT_full, size_t(dim) * size_t(db), 67890u);
    CUDA_CHECK(cudaGetLastError());
  }

  // cuBLAS
  cublasHandle_t handle{};
  CUBLAS_CHECK(cublasCreate(&handle));
  if (tf32) {
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  auto run_once = [&](bool timed) -> float {
    // init best
    {
      int threads = 256;
      int blocks = (nq + threads - 1) / threads;
      init_best<<<blocks, threads>>>(dBest, dBestIdx, nq);
      CUDA_CHECK(cudaGetLastError());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    if (timed) CUDA_CHECK(cudaEventRecord(start));

    const float alpha = 1.0f, beta = 0.0f;

    long long processed = 0;
    while (processed < db) {
      int this_chunk = (int)std::min<long long>(chunk, db - processed);

      const float* dDBT = nullptr;
      if (resident) {
        dDBT = dDBT_full + processed * (long long)dim; // column offset
      } else {
        // Fill chunk buffer deterministically
        size_t n = size_t(dim) * size_t(this_chunk);
        fill_random_flat<<<fill_blocks, fill_threads>>>(dDBT_chunk, n, 67890u + (uint32_t)processed);
        CUDA_CHECK(cudaGetLastError());
        dDBT = dDBT_chunk;
      }

      // SGEMM: S(nq x this_chunk) = Q(nq x dim) * DBT(dim x this_chunk)
      // Column-major (cuBLAS default), leading dims: ldQ=nq, ldDBT=dim, ldS=nq
      CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        nq, this_chunk, dim,
        &alpha,
        dQ, nq,
        dDBT, dim,
        &beta,
        dS, nq
      ));

      // Reduce per query
      // Launch: one block per query, 256 threads
      reduce_row_max_update<<<nq, 256>>>(dS, nq, this_chunk, dBest, dBestIdx, (int)processed);
      CUDA_CHECK(cudaGetLastError());

      processed += this_chunk;
    }

    if (timed) {
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
    } else {
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    float ms = 0.0f;
    if (timed) CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
  };

  // Warmup
  for (int i=0; i<warmup; i++) (void)run_once(false);

  // Timed runs
  std::vector<float> times_ms;
  times_ms.reserve(runs);
  for (int i=0; i<runs; i++) times_ms.push_back(run_once(true));

  // Stats
  auto sorted = times_ms;
  std::sort(sorted.begin(), sorted.end());
  float min_ms = sorted.front();
  float med_ms = sorted[sorted.size()/2];
  float max_ms = sorted.back();
  double avg_ms = 0.0;
  for (float t : times_ms) avg_ms += t;
  avg_ms /= times_ms.size();

  // Derived metrics (GEMM FLOP count only)
  double flops = 2.0 * double(nq) * double(db) * double(dim); // multiply-add ~2 flops
  double avg_s = (avg_ms / 1000.0);
  double tflops = (flops / avg_s) / 1e12;

  double comps = double(nq) * double(db);
  double comps_per_s = comps / avg_s;

  // Report
  print_line();
  std::printf("gvsbench  |  GPU Vector Search Benchmark (Inner Product Top-1)\n");
  print_line();
  std::printf("GPU               : #%d  %s\n", device, prop.name);
  std::printf("Compute Capability : %d.%d\n", prop.major, prop.minor);
  std::printf("SMs               : %d\n", prop.multiProcessorCount);
  std::printf("Global Memory      : %s\n", bytes_human((double)prop.totalGlobalMem).c_str());
  std::printf("Free Memory (now)  : %s\n", bytes_human((double)freeB).c_str());
  std::printf("cuBLAS Math Mode   : %s\n", tf32 ? "TF32 Tensor Op (if supported)" : "Default (FP32)");
  std::printf("DB Residency       : %s\n", resident ? "Resident in VRAM" : "Streaming (chunk buffer)");
  print_line();
  std::printf("Parameters\n");
  std::printf("  db vectors       : %lld\n", db);
  std::printf("  queries          : %d\n", nq);
  std::printf("  dim              : %d\n", dim);
  std::printf("  chunk            : %d\n", chunk);
  std::printf("  warmup / runs    : %d / %d\n", warmup, runs);
  print_line();
  std::printf("Results (ms per full DB sweep)\n");
  std::printf("  min / median / max : %.3f / %.3f / %.3f ms\n", min_ms, med_ms, max_ms);
  std::printf("  avg                : %.3f ms\n", avg_ms);
  print_line();
  std::printf("Throughput (derived)\n");
  std::printf("  comparisons/sec     : %.3e  (query-db dot products / sec)\n", comps_per_s);
  std::printf("  GEMM TFLOP/s (est.) : %.3f\n", tflops);
  print_line();

  // Cleanup
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dQ));
  CUDA_CHECK(cudaFree(dS));
  CUDA_CHECK(cudaFree(dBest));
  CUDA_CHECK(cudaFree(dBestIdx));
  if (dDBT_full) CUDA_CHECK(cudaFree(dDBT_full));
  if (dDBT_chunk) CUDA_CHECK(cudaFree(dDBT_chunk));

  return 0;
}
