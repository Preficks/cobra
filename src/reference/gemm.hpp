// NOTE: WRITTEN BY OPUS 5.5, NOT PERSONALLY WRITTEN

#pragma once

// Multithreaded, cache-blocked GEMM engine backing matmul_optimized and contract_optimized.
//
// Structure follows the BLIS/GotoBLAS design:
//   jc (NC cols of B)  ->  pc (KC depth)  ->  ic (MC rows of A)
//     pack B[pc:pc+KC, jc:jc+NC] into NR-wide panels   (stays in L2, shared by all threads)
//     pack A[ic:ic+MC, pc:pc+KC] into MR-tall panels   (stays in L2, shared by all threads)
//     parallel over 2D groups of micro-tiles; each micro-tile is an MR x NR register-blocked
//     NEON kernel that streams one A panel and one B panel (B panel stays in L1).
//
// Packing reads its operands through a `layout`, so any single-index tensor contraction
// (not just row-major matmul) runs through the same kernels without first transposing.
//
// Opt-in: define TENSOR_USE_ACCELERATE (and link -framework Accelerate) to route float/double
// through Apple's cblas_?gemm, which runs on the AMX matrix unit (~1.6x the NEON peak on M2).
// Shapes BLAS cannot express efficiently, and every other element type, stay on NEON.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(TENSOR_USE_ACCELERATE)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

namespace details::opt {

// Element (i, k) of an operand lives at  (i / div) * outer + (i % div) + k * kstride.
// i is the free (row/col) index, k the contracted index. Row-major A[N,K]: {1, K, 1}.
// Row-major B[K,P] indexed by column j: {P, K * P, P}.
struct layout {
  std::size_t div, outer, kstride;
  constexpr std::size_t base(std::size_t i) const { return (i / div) * outer + (i % div); }
};

// Below this many multiply-adds a plain loop beats packing / BLAS call overhead.
template <class T>
inline constexpr std::size_t direct_threshold =
#if defined(TENSOR_USE_ACCELERATE)
    std::is_same_v<T, float> || std::is_same_v<T, double> ? 16 * 16 * 16 :
#endif
    32 * 32 * 32;

constexpr std::size_t ceil_div(std::size_t x, std::size_t m) { return (x + m - 1) / m; }
constexpr std::size_t round_up(std::size_t x, std::size_t m) { return ceil_div(x, m) * m; }

// ---------------------------------------------------------------------------------------
// Threading: libdispatch on Apple (persistent pool, dynamic scheduling, P+E cores),
// a plain atomic work queue over std::thread elsewhere.
// ---------------------------------------------------------------------------------------

// Fast cores (P-cores on Apple silicon) and all cores.
inline std::size_t fast_cores() {
  static const std::size_t n = [] {
#if defined(__APPLE__)
    int v = 0;
    std::size_t len = sizeof v;
    if (sysctlbyname("hw.perflevel0.logicalcpu", &v, &len, nullptr, 0) == 0 && v > 0) return std::size_t(v);
#endif
    return std::size_t(std::max(1u, std::thread::hardware_concurrency()));
  }();
  return n;
}

inline std::size_t all_cores() {
  static const std::size_t n = std::max(1u, std::thread::hardware_concurrency());
  return n;
}

// Runs f(0..n-1) on up to `workers` threads that pull indices from a shared counter,
// so slower cores simply end up taking fewer tasks.
template <class F>
void parallel_for(std::size_t n, std::size_t workers, F&& f) {
  workers = std::min(workers, n);
  if (workers <= 1) {
    for (std::size_t i = 0; i < n; ++i) f(i);
    return;
  }
  struct ctx_t {
    std::remove_reference_t<F>* f;
    std::size_t n;
    std::atomic<std::size_t> next{0};
  } ctx{&f, n};
  auto worker = [](void* c, std::size_t) {
    auto& x = *static_cast<ctx_t*>(c);
    for (std::size_t i; (i = x.next.fetch_add(1, std::memory_order_relaxed)) < x.n;) (*x.f)(i);
  };
#if defined(__APPLE__)
  dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), &ctx, worker);
#else
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  for (std::size_t t = 1; t < workers; ++t) pool.emplace_back(worker, &ctx, t);
  worker(&ctx, 0);
  for (auto& t : pool) t.join();
#endif
}

// ---------------------------------------------------------------------------------------
// Micro-kernels: C[MR x NR] (+)= Ap[kc x MR]^T * Bp[kc x NR]
// Ap holds MR contiguous values per k, Bp holds NR contiguous values per k.
// ---------------------------------------------------------------------------------------

template <class T>
struct kernel_traits {
  static constexpr std::size_t MR = 4, NR = 8;
  static constexpr std::size_t KC = 256, MC = 256, NC = 2048;
};

#if defined(__ARM_NEON) && defined(__aarch64__)
// 6x8 doubles: 24 accumulator q-registers + 4 for B + 3 for A = 31 of 32.
template <>
struct kernel_traits<double> {
  static constexpr std::size_t MR = 6, NR = 8;
  static constexpr std::size_t KC = 256, MC = 504, NC = 2048;
};

// 8x12 floats: 24 accumulators + 3 for B + 2 for A.
template <>
struct kernel_traits<float> {
  static constexpr std::size_t MR = 8, NR = 12;
  static constexpr std::size_t KC = 512, MC = 512, NC = 2040;
};
#endif

// Writes an MR x NR register tile (staged in `tile`) to C, handling ragged edges.
template <class T, std::size_t MR, std::size_t NR>
inline void store_edge(const T (&tile)[MR][NR], T* C, std::size_t ldc, bool accumulate,
                       std::size_t mr, std::size_t nr) {
  for (std::size_t i = 0; i < mr; ++i)
    for (std::size_t j = 0; j < nr; ++j)
      C[i * ldc + j] = accumulate ? C[i * ldc + j] + tile[i][j] : tile[i][j];
}

template <class T>
inline void kernel(std::size_t kc, const T* __restrict Ap, const T* __restrict Bp,
                   T* __restrict C, std::size_t ldc, bool accumulate, std::size_t mr, std::size_t nr) {
  constexpr std::size_t MR = kernel_traits<T>::MR, NR = kernel_traits<T>::NR;
  T acc[MR][NR] = {};
  for (std::size_t k = 0; k < kc; ++k, Ap += MR, Bp += NR)
    for (std::size_t i = 0; i < MR; ++i)
      for (std::size_t j = 0; j < NR; ++j)
        acc[i][j] += Ap[i] * Bp[j];
  store_edge<T, MR, NR>(acc, C, ldc, accumulate, mr, nr);
}

#if defined(__ARM_NEON) && defined(__aarch64__)
template <>
inline void kernel<double>(std::size_t kc, const double* __restrict Ap, const double* __restrict Bp,
                           double* __restrict C, std::size_t ldc, bool accumulate,
                           std::size_t mr, std::size_t nr) {
  constexpr std::size_t MR = 6, NR = 8;
  float64x2_t c[MR][4];
  for (auto& row : c)
    for (auto& v : row) v = vdupq_n_f64(0.0);

#define ROW(i, av, lane)                                \
  c[i][0] = vfmaq_laneq_f64(c[i][0], b0, av, lane);     \
  c[i][1] = vfmaq_laneq_f64(c[i][1], b1, av, lane);     \
  c[i][2] = vfmaq_laneq_f64(c[i][2], b2, av, lane);     \
  c[i][3] = vfmaq_laneq_f64(c[i][3], b3, av, lane);

#pragma clang loop unroll_count(4)
  for (std::size_t k = 0; k < kc; ++k, Ap += MR, Bp += NR) {
    const float64x2_t b0 = vld1q_f64(Bp), b1 = vld1q_f64(Bp + 2);
    const float64x2_t b2 = vld1q_f64(Bp + 4), b3 = vld1q_f64(Bp + 6);
    const float64x2_t a01 = vld1q_f64(Ap), a23 = vld1q_f64(Ap + 2), a45 = vld1q_f64(Ap + 4);
    ROW(0, a01, 0) ROW(1, a01, 1)
    ROW(2, a23, 0) ROW(3, a23, 1)
    ROW(4, a45, 0) ROW(5, a45, 1)
  }
#undef ROW

  if (mr == MR && nr == NR) {
    for (std::size_t i = 0; i < MR; ++i) {
      double* p = C + i * ldc;
      for (std::size_t v = 0; v < 4; ++v) {
        const float64x2_t r = accumulate ? vaddq_f64(vld1q_f64(p + 2 * v), c[i][v]) : c[i][v];
        vst1q_f64(p + 2 * v, r);
      }
    }
  } else {
    double tile[MR][NR];
    for (std::size_t i = 0; i < MR; ++i)
      for (std::size_t v = 0; v < 4; ++v) vst1q_f64(&tile[i][2 * v], c[i][v]);
    store_edge<double, MR, NR>(tile, C, ldc, accumulate, mr, nr);
  }
}

template <>
inline void kernel<float>(std::size_t kc, const float* __restrict Ap, const float* __restrict Bp,
                          float* __restrict C, std::size_t ldc, bool accumulate,
                          std::size_t mr, std::size_t nr) {
  constexpr std::size_t MR = 8, NR = 12;
  float32x4_t c[MR][3];
  for (auto& row : c)
    for (auto& v : row) v = vdupq_n_f32(0.0f);

#define ROW(i, av, lane)                                \
  c[i][0] = vfmaq_laneq_f32(c[i][0], b0, av, lane);     \
  c[i][1] = vfmaq_laneq_f32(c[i][1], b1, av, lane);     \
  c[i][2] = vfmaq_laneq_f32(c[i][2], b2, av, lane);

#pragma clang loop unroll_count(4)
  for (std::size_t k = 0; k < kc; ++k, Ap += MR, Bp += NR) {
    const float32x4_t b0 = vld1q_f32(Bp), b1 = vld1q_f32(Bp + 4), b2 = vld1q_f32(Bp + 8);
    const float32x4_t a0 = vld1q_f32(Ap), a1 = vld1q_f32(Ap + 4);
    ROW(0, a0, 0) ROW(1, a0, 1) ROW(2, a0, 2) ROW(3, a0, 3)
    ROW(4, a1, 0) ROW(5, a1, 1) ROW(6, a1, 2) ROW(7, a1, 3)
  }
#undef ROW

  if (mr == MR && nr == NR) {
    for (std::size_t i = 0; i < MR; ++i) {
      float* p = C + i * ldc;
      for (std::size_t v = 0; v < 3; ++v) {
        const float32x4_t r = accumulate ? vaddq_f32(vld1q_f32(p + 4 * v), c[i][v]) : c[i][v];
        vst1q_f32(p + 4 * v, r);
      }
    }
  } else {
    float tile[MR][NR];
    for (std::size_t i = 0; i < MR; ++i)
      for (std::size_t v = 0; v < 3; ++v) vst1q_f32(&tile[i][4 * v], c[i][v]);
    store_edge<float, MR, NR>(tile, C, ldc, accumulate, mr, nr);
  }
}
#endif

// ---------------------------------------------------------------------------------------
// Packing. A panel: for each k, MR consecutive rows. B panel: for each k, NR consecutive cols.
// Rows/cols past the edge are zero-filled so the kernel never branches.
// ---------------------------------------------------------------------------------------

template <class T, std::size_t R, layout L>
inline void pack_panel(const T* X, std::size_t i0, std::size_t r, std::size_t pc, std::size_t kc,
                       T* __restrict out) {
  // Contiguous run along the free index (e.g. a row of row-major B): straight copies.
  if (r == R && (i0 % L.div) + R <= L.div) {
    const T* src = X + L.base(i0) + pc * L.kstride;
    for (std::size_t k = 0; k < kc; ++k, out += R) std::memcpy(out, src + k * L.kstride, R * sizeof(T));
    return;
  }
  const T* src[R];
  for (std::size_t i = 0; i < R; ++i) src[i] = X + L.base(i0 + std::min(i, r - 1)) + pc * L.kstride;
  if (r == R) {
    for (std::size_t k = 0; k < kc; ++k, out += R)
      for (std::size_t i = 0; i < R; ++i) out[i] = src[i][k * L.kstride];
  } else {
    for (std::size_t k = 0; k < kc; ++k, out += R)
      for (std::size_t i = 0; i < R; ++i) out[i] = i < r ? src[i][k * L.kstride] : T{};
  }
}

template <class T>
T* scratch(std::vector<T>& buf, std::size_t n) {
  if (buf.size() < n) buf.resize(n);
  return buf.data();
}

#if defined(TENSOR_USE_ACCELERATE)
// ---------------------------------------------------------------------------------------
// Accelerate backend. A layout splits its free index into `blocks` runs of `len` elements;
// within a run, element (i, k) sits at i * s_free + k * s_k with one of the strides == 1,
// which is a (possibly transposed) BLAS matrix. Contracting a middle axis gives several runs:
// those are gathered into one contiguous matrix with memcpy (O(operand) work, vs the O(N*K*P)
// multiply) so AMX always sees a single large GEMM rather than many narrow ones.
// ---------------------------------------------------------------------------------------

struct blas_view {
  std::size_t blocks, len, block_stride, s_free, s_k;
};

constexpr blas_view as_blas(layout L, std::size_t count) {
  if (L.div == 1) return {1, count, 0, L.outer, L.kstride};
  return {count / L.div, L.div, L.outer, 1, L.kstride};
}

inline void blas_gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, std::size_t m, std::size_t n, std::size_t k,
                      const double* A, std::size_t lda, const double* B, std::size_t ldb,
                      double* C, std::size_t ldc) {
  cblas_dgemm(CblasRowMajor, ta, tb, __LAPACK_int(m), __LAPACK_int(n), __LAPACK_int(k), 1.0,
              A, __LAPACK_int(lda), B, __LAPACK_int(ldb), 0.0, C, __LAPACK_int(ldc));
}

inline void blas_gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, std::size_t m, std::size_t n, std::size_t k,
                      const float* A, std::size_t lda, const float* B, std::size_t ldb,
                      float* C, std::size_t ldc) {
  cblas_sgemm(CblasRowMajor, ta, tb, __LAPACK_int(m), __LAPACK_int(n), __LAPACK_int(k), 1.0f,
              A, __LAPACK_int(lda), B, __LAPACK_int(ldb), 0.0f, C, __LAPACK_int(ldc));
}

// Gathers a multi-run operand into a contiguous [K x count] matrix (k-major, free index fastest).
template <class T>
const T* gather_runs(const blas_view& v, std::size_t K, const T* X, std::vector<T>& buf) {
  T* out = scratch(buf, K * v.blocks * v.len);
  const std::size_t ld = v.blocks * v.len;
  for (std::size_t r = 0; r < v.blocks; ++r)
    for (std::size_t k = 0; k < K; ++k)
      std::memcpy(out + k * ld + r * v.len, X + r * v.block_stride + k * v.s_k, v.len * sizeof(T));
  return out;
}

template <class T, layout LA, layout LB>
void accelerate_gemm(std::size_t N, std::size_t K, std::size_t P, const T* A, const T* B, T* C) {
  const blas_view a = as_blas(LA, N), b = as_blas(LB, P);
  thread_local std::vector<T> a_buf, b_buf;

  CBLAS_TRANSPOSE ta, tb;
  std::size_t lda, ldb;
  if (a.blocks > 1) {
    A = gather_runs(a, K, A, a_buf);
    ta = CblasTrans, lda = N;
  } else if (a.s_k == 1) {
    ta = CblasNoTrans, lda = a.s_free;   // rows contiguous along k
  } else {
    ta = CblasTrans, lda = a.s_k;        // stored k-major
  }
  if (b.blocks > 1) {
    B = gather_runs(b, K, B, b_buf);
    tb = CblasNoTrans, ldb = P;
  } else if (b.s_free == 1) {
    tb = CblasNoTrans, ldb = b.s_k;      // rows of B contiguous along j
  } else {
    tb = CblasTrans, ldb = b.s_free;     // stored j-major
  }
  blas_gemm(ta, tb, N, P, K, A, lda, B, ldb, C, P);
}
#endif

// ---------------------------------------------------------------------------------------
// C[N x P] = A[N x K] * B[K x P], C row-major with ldc = P, A and B read through LA / LB.
// ---------------------------------------------------------------------------------------

template <class T, layout LA, layout LB>
void gemm(std::size_t N, std::size_t K, std::size_t P, const T* A, const T* B, T* C) {
  using KT = kernel_traits<T>;
  constexpr std::size_t MR = KT::MR, NR = KT::NR, KC = KT::KC, MC = KT::MC, NC = KT::NC;
  // Micro-tiles per parallel task (rows of MR panels x cols of NR panels).
  constexpr std::size_t GR = 8, GC = 4;
  constexpr std::size_t PACK_CHUNK = 8;

  if (K == 0) {
    std::fill(C, C + N * P, T{});
    return;
  }

#if defined(TENSOR_USE_ACCELERATE)
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    accelerate_gemm<T, LA, LB>(N, K, P, A, B, C);
    return;
  }
#endif

  // Small problems: dispatch wakeups cost more than they save, and below ~256^3 the
  // efficiency cores tend to finish last and stall the fast ones, so leave them out.
  const double work = static_cast<double>(N) * K * P;
  const std::size_t workers = work < 80.0 * 80.0 * 80.0   ? 1
                            : work < 256.0 * 256.0 * 256.0 ? fast_cores()
                                                           : all_cores();

  thread_local std::vector<T> a_buf, b_buf;
  T* Ap = scratch(a_buf, round_up(std::min(MC, N), MR) * std::min(KC, K));
  T* Bp = scratch(b_buf, round_up(std::min(NC, P), NR) * std::min(KC, K));

  for (std::size_t jc = 0; jc < P; jc += NC) {
    const std::size_t nc = std::min(NC, P - jc);
    const std::size_t nb = ceil_div(nc, NR);

    for (std::size_t pc = 0; pc < K; pc += KC) {
      const std::size_t kc = std::min(KC, K - pc);
      const bool accumulate = pc != 0;

      for (std::size_t ic = 0; ic < N; ic += MC) {
        const std::size_t mc = std::min(MC, N - ic);
        const std::size_t na = ceil_div(mc, MR);

        // Pack A (and B on the first row block) in one parallel phase.
        const std::size_t a_jobs = ceil_div(na, PACK_CHUNK);
        const std::size_t b_jobs = ic == 0 ? ceil_div(nb, PACK_CHUNK) : 0;
        parallel_for(a_jobs + b_jobs, workers, [&](std::size_t job) {
          if (job < a_jobs) {
            for (std::size_t p = job * PACK_CHUNK; p < std::min(na, (job + 1) * PACK_CHUNK); ++p) {
              const std::size_t i0 = p * MR;
              pack_panel<T, MR, LA>(A, ic + i0, std::min(MR, mc - i0), pc, kc, Ap + p * kc * MR);
            }
          } else {
            job -= a_jobs;
            for (std::size_t p = job * PACK_CHUNK; p < std::min(nb, (job + 1) * PACK_CHUNK); ++p) {
              const std::size_t j0 = p * NR;
              pack_panel<T, NR, LB>(B, jc + j0, std::min(NR, nc - j0), pc, kc, Bp + p * kc * NR);
            }
          }
        });

        const std::size_t tr = ceil_div(na, GR), tc = ceil_div(nb, GC);
        parallel_for(tr * tc, workers, [&](std::size_t t) {
          const std::size_t r0 = (t / tc) * GR, c0 = (t % tc) * GC;
          for (std::size_t jp = c0; jp < std::min(nb, c0 + GC); ++jp) {
            const std::size_t j0 = jp * NR, nr = std::min(NR, nc - j0);
            const T* bp = Bp + jp * kc * NR;
            for (std::size_t ip = r0; ip < std::min(na, r0 + GR); ++ip) {
              const std::size_t i0 = ip * MR, mr = std::min(MR, mc - i0);
              kernel<T>(kc, Ap + ip * kc * MR, bp, C + (ic + i0) * P + jc + j0, P, accumulate, mr, nr);
            }
          }
        });
      }
    }
  }
}

} // namespace details::opt
