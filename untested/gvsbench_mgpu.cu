// gvsbench_mgpu.cu
// Warning: Not yet tested - Please use with caution. Shown to the public for those who want to experiment on multi-GPU environment.
// Multi-GPU Vector Search Benchmark (Inner Product Top-1) using cuBLAS SGEMM + reduction.
// - Shards the DB across GPUs.
// - Replicates Q on each GPU.
// - Runs one host thread per GPU.
// - Reports aggregate time as max(per-GPU time) per sweep.
//
// Compile:
//   nvcc -O3 -std=c++17 gvsbench_mgpu.cu -lcublas -o gvsbench_mgpu

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
#include <thread>
#include <mutex>
#include <sstream>
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

// Hash callable from device code
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
// Grid-stride loop avoids overflow / huge grids.
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

// Reduce row-wise max for S of shape (nq x chunk), column-major (ld=nq)
// Updates best_scores / best_idx with a global DB offset.
__global__ void reduce_row_max_update(
    const float* __restrict__ S,
    int nq, int chunk,
    float* __restrict__ best_scores,
    int* __restrict__ best_idx,
    int global_db_offset)
{
  int q = blockIdx.x; // one block per query
  if (q >= nq) return;

  float local_max = -INFINITY;
  int local_arg = -1;

  for (int j = threadIdx.x; j < chunk; j += blockDim.x) {
    float v = S[q + nq * j];
    if (v > local_max) { local_max = v; local_arg = j; }
  }

  // Fixed for 256 threads per block
  __shared__ float smax[256];
  __shared__ int   sarg[256];

  int t = threadIdx.x;
  smax[t] = local_max;
  sarg[t] = local_arg;
  __syncthreads();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (t < stride) {
      float v2 = smax[t + stride];
      int a2 = sarg[t + stride];
      if (v2 > smax[t]) { smax[t] = v2; sarg[t] = a2; }
    }
    __syncthreads();
  }

  if (t == 0) {
    float prev = best_scores[q];
    float cand = smax[0];
    if (cand > prev) {
      best_scores[q] = cand;
      best_idx[q] = global_db_offset + sarg[0];
    }
  }
}

static void print_line(int w=90) {
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

static std::vector<int> parse_gpu_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(std::stoi(item));
  }
  return out;
}

static void usage() {
  std::printf(
R"(gvsbench_mgpu - Multi-GPU Vector Search Benchmark (CUDA/cuBLAS)

Usage:
  ./gvsbench_mgpu [options]

Options:
  --gpus LIST         comma-separated GPU list (e.g. 0,1,2). Default: all GPUs
  --db N              number of database vectors (default: 262144)
  --queries N          number of query vectors (default: 1024)
  --dim N             vector dimension (default: 256)
  --chunk N           DB chunk size per GEMM (default: 16384)
  --runs N            timed runs (default: 10)
  --warmup N          warmup runs (default: 2)
  --tf32 0|1          enable TF32 math in cuBLAS (default: 0)
  --resident -1|0|1   -1 auto (default), 0 force streaming, 1 force resident shard in VRAM
  --help              show help

What it does:
  Finds per-query top-1 by inner product:
    best[q] = max_i dot(Q[q], X[i])
  Multi-GPU: shards the DB across GPUs and runs each shard concurrently.
)");
}

struct DeviceCtx {
  int dev = 0;
  cudaDeviceProp prop{};

  // DB shard range in global index space: [db_begin, db_end)
  long long db_begin = 0;
  long long db_end = 0;
  long long db_size = 0;

  int nq = 0;
  int dim = 0;
  int chunk = 0;

  bool resident = false;
  bool tf32 = false;

  // GPU resources
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;

  float* dQ = nullptr;        // (nq x dim) col-major
  float* dS = nullptr;        // (nq x chunk) col-major
  float* dBest = nullptr;     // (nq)
  int*   dBestIdx = nullptr;  // (nq)

  // DB^T (dim x db_size) if resident, else chunk buffer (dim x chunk)
  float* dDBT_resident = nullptr;
  float* dDBT_chunk = nullptr;

  // Fill kernel config
  int fill_threads = 256;
  int fill_blocks = 0;

  void setup(int device_id,
             long long begin, long long end,
             int nq_, int dim_, int chunk_,
             bool tf32_, int resident_flag /* -1/0/1 */)
  {
    dev = device_id;
    db_begin = begin;
    db_end = end;
    db_size = db_end - db_begin;
    nq = nq_;
    dim = dim_;
    chunk = chunk_;
    tf32 = tf32_;

    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Decide residency per shard
    size_t freeB=0,totalB=0;
    CUDA_CHECK(cudaMemGetInfo(&freeB,&totalB));

    size_t bytesQ = size_t(nq) * size_t(dim) * sizeof(float);
    size_t bytesS = size_t(nq) * size_t(chunk) * sizeof(float);
    size_t bytesBest = size_t(nq) * (sizeof(float) + sizeof(int));
    size_t bytesDBFull = size_t(dim) * size_t(db_size) * sizeof(float);

    if (resident_flag == 1) resident = true;
    else if (resident_flag == 0) resident = false;
    else {
      double need = double(bytesQ + bytesS + bytesBest + bytesDBFull);
      resident = (need < double(freeB) * 0.70);
    }
    if (resident && (bytesQ + bytesS + bytesBest + bytesDBFull > freeB)) {
      resident = false;
    }

    // stream + cuBLAS
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    if (tf32) {
      CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }

    // memory alloc
    CUDA_CHECK(cudaMalloc(&dQ, bytesQ));
    CUDA_CHECK(cudaMalloc(&dS, bytesS));
    CUDA_CHECK(cudaMalloc(&dBest, nq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBestIdx, nq * sizeof(int)));

    if (resident) {
      CUDA_CHECK(cudaMalloc(&dDBT_resident, bytesDBFull));
    } else {
      CUDA_CHECK(cudaMalloc(&dDBT_chunk, size_t(dim) * size_t(chunk) * sizeof(float)));
    }

    fill_blocks = std::min(prop.multiProcessorCount * 8, 65535);

    // Fill Q (same seed on all GPUs so results are consistent)
    fill_random_flat<<<fill_blocks, fill_threads, 0, stream>>>(
      dQ, size_t(nq) * size_t(dim), 12345u
    );
    CUDA_CHECK(cudaGetLastError());

    // Fill resident DB shard once (DB^T layout dim x db_size)
    if (resident) {
      // seed includes db_begin so shards are distinct but deterministic
      uint32_t seed = 67890u ^ uint32_t(db_begin);
      fill_random_flat<<<fill_blocks, fill_threads, 0, stream>>>(
        dDBT_resident, size_t(dim) * size_t(db_size), seed
      );
      CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Returns elapsed ms for one sweep over this shard.
  // NOTE: if GPUs run concurrently, aggregate sweep time is max over devices.
  float run_once(bool timed) {
    CUDA_CHECK(cudaSetDevice(dev));

    // init best
    int init_threads = 256;
    int init_blocks = (nq + init_threads - 1) / init_threads;
    init_best<<<init_blocks, init_threads, 0, stream>>>(dBest, dBestIdx, nq);
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    if (timed) CUDA_CHECK(cudaEventRecord(start, stream));

    const float alpha = 1.0f, beta = 0.0f;

    long long processed = 0;
    while (processed < db_size) {
      int this_chunk = (int)std::min<long long>(chunk, db_size - processed);

      const float* dDBT = nullptr;

      if (resident) {
        dDBT = dDBT_resident + processed * (long long)dim; // column offset
      } else {
        // Fill chunk buffer each iteration (dim x this_chunk)
        uint32_t seed = (67890u ^ uint32_t(db_begin)) + uint32_t(processed);
        fill_random_flat<<<fill_blocks, fill_threads, 0, stream>>>(
          dDBT_chunk, size_t(dim) * size_t(this_chunk), seed
        );
        CUDA_CHECK(cudaGetLastError());
        dDBT = dDBT_chunk;
      }

      // SGEMM: S(nq x this_chunk) = Q(nq x dim) * DBT(dim x this_chunk)
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

      // Reduce: one block per query row, 256 threads
      // global offset = db_begin + processed
      reduce_row_max_update<<<nq, 256, 0, stream>>>(
        dS, nq, this_chunk,
        dBest, dBestIdx,
        (int)(db_begin + processed)
      );
      CUDA_CHECK(cudaGetLastError());

      processed += this_chunk;
    }

    if (timed) {
      CUDA_CHECK(cudaEventRecord(stop, stream));
      CUDA_CHECK(cudaEventSynchronize(stop));
    } else {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    float ms = 0.0f;
    if (timed) CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
  }

  void destroy() {
    CUDA_CHECK(cudaSetDevice(dev));
    if (dQ) CUDA_CHECK(cudaFree(dQ));
    if (dS) CUDA_CHECK(cudaFree(dS));
    if (dBest) CUDA_CHECK(cudaFree(dBest));
    if (dBestIdx) CUDA_CHECK(cudaFree(dBestIdx));
    if (dDBT_resident) CUDA_CHECK(cudaFree(dDBT_resident));
    if (dDBT_chunk) CUDA_CHECK(cudaFree(dDBT_chunk));
    if (handle) CUBLAS_CHECK(cublasDestroy(handle));
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
  }
};

int main(int argc, char** argv) {
  std::string gpus_arg;
  long long db = 262144;
  int nq = 1024;
  int dim = 256;
  int chunk = 16384;
  int runs = 10;
  int warmup = 2;
  int tf32 = 0;
  int resident_flag = -1;

  static option long_opts[] = {
    {"gpus",     required_argument, nullptr, 0},
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
    if (opt == "gpus") gpus_arg = optarg;
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

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::fprintf(stderr, "No CUDA devices found.\n");
    return 1;
  }

  std::vector<int> gpus;
  if (!gpus_arg.empty()) {
    gpus = parse_gpu_list(gpus_arg);
  } else {
    gpus.resize(device_count);
    for (int i=0;i<device_count;i++) gpus[i] = i;
  }

  // Validate GPUs
  for (int d : gpus) {
    if (d < 0 || d >= device_count) {
      std::fprintf(stderr, "Invalid GPU index in --gpus: %d (device_count=%d)\n", d, device_count);
      return 1;
    }
  }

  int ng = (int)gpus.size();
  if (ng <= 0) {
    std::fprintf(stderr, "No GPUs selected.\n");
    return 1;
  }

  // Shard DB across GPUs evenly: contiguous ranges
  // GPU k gets [floor(k*db/ng), floor((k+1)*db/ng))
  std::vector<DeviceCtx> ctxs(ng);

  for (int k=0;k<ng;k++) {
    long long begin = (db * k) / ng;
    long long end   = (db * (k+1)) / ng;
    long long sz = end - begin;
    if (sz <= 0) { begin = end = 0; sz = 0; }

    int dev = gpus[k];
    ctxs[k].setup(dev, begin, end, nq, dim, std::min<long long>(chunk, sz > 0 ? sz : 1),
                  (tf32 != 0), resident_flag);
  }

  // Print header + per-GPU mapping
  print_line();
  std::printf("gvsbench_mgpu | Multi-GPU Vector Search Benchmark (Inner Product Top-1)\n");
  print_line();
  std::printf("Selected GPUs: ");
  for (int i=0;i<ng;i++) std::printf("%d%s", gpus[i], (i+1<ng?",":""));
  std::printf("\n");
  std::printf("Parameters: db=%lld, queries=%d, dim=%d, chunk=%d, warmup=%d, runs=%d, tf32=%d, resident=%d\n",
              db, nq, dim, chunk, warmup, runs, tf32, resident_flag);
  print_line();
  std::printf("DB sharding (global indices):\n");
  for (int k=0;k<ng;k++) {
    auto& c = ctxs[k];
    std::printf("  GPU #%d %-28s  shard=[%lld, %lld)  vectors=%lld  mode=%s\n",
                c.dev, c.prop.name, c.db_begin, c.db_end, c.db_size,
                c.resident ? "resident" : "stream");
  }
  print_line();

  auto run_parallel = [&](bool timed) -> float {
    std::vector<float> ms(ng, 0.0f);
    std::vector<std::thread> threads;
    threads.reserve(ng);

    for (int k=0;k<ng;k++) {
      threads.emplace_back([&, k]() {
        ms[k] = ctxs[k].run_once(timed);
      });
    }
    for (auto& t : threads) t.join();

    // Aggregate as the max GPU time (they should overlap if the host has enough threads)
    return *std::max_element(ms.begin(), ms.end());
  };

  // Warmup sweeps
  for (int i=0;i<warmup;i++) (void)run_parallel(false);

  // Timed sweeps
  std::vector<float> times_ms;
  times_ms.reserve(runs);
  for (int i=0;i<runs;i++) {
    times_ms.push_back(run_parallel(true));
  }

  // Stats
  auto sorted = times_ms;
  std::sort(sorted.begin(), sorted.end());
  float min_ms = sorted.front();
  float med_ms = sorted[sorted.size()/2];
  float max_ms = sorted.back();
  double avg_ms = 0.0;
  for (float t : times_ms) avg_ms += t;
  avg_ms /= times_ms.size();

  // Derived metrics (GEMM FLOP count across ALL comparisons)
  // Total FLOPs per sweep: 2 * nq * db * dim
  double flops = 2.0 * double(nq) * double(db) * double(dim);
  double avg_s = (avg_ms / 1000.0);
  double tflops = (flops / avg_s) / 1e12;

  double comps = double(nq) * double(db);
  double comps_per_s = comps / avg_s;

  // Output
  std::printf("Results (ms per full DB sweep, aggregate=max over GPUs)\n");
  std::printf("  min / median / max : %.3f / %.3f / %.3f ms\n", min_ms, med_ms, max_ms);
  std::printf("  avg                : %.3f ms\n", avg_ms);
  print_line();
  std::printf("Throughput (derived, aggregate)\n");
  std::printf("  comparisons/sec     : %.3e  (query-db dot products / sec)\n", comps_per_s);
  std::printf("  GEMM TFLOP/s (est.) : %.3f\n", tflops);
  print_line();

  // Cleanup
  for (auto& c : ctxs) c.destroy();
  return 0;
}
