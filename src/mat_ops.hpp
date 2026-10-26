#include "config.hpp"
#include "tensor.hpp"


template <Numeric T, size_t N, size_t M>
[[nodiscard]] constexpr matrix<T, M, N> transpose(const matrix<T, N, M>& m) {
  matrix<T, M, N> res{};
  for(size_t i{}; i < N; ++i) {
    for(size_t j{}; j < M; ++j) {
      res[j, i] = m[i, j];
    }
  }

  return res;
}

template <Numeric T, std::size_t N, std::size_t M, std::size_t P>
[[nodiscard]] constexpr matrix<T, N, P> matmul(const matrix<T, N, M>& m1, const matrix<T, M, P>& m2) {
    matrix<T, N, P> res{};

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < M; ++k) {
            const T a = m1[i, k];
            for (std::size_t j = 0; j < P; ++j) {
                res[i, j] += a * m2[k, j];
            }
        }
    }

    return res;
}


template <typename T, std::size_t MR, std::size_t NR>
inline void micro_kernel(std::size_t K,
                         const T* __restrict A, std::size_t lda,
                         const T* __restrict B, std::size_t ldb,
                         T* __restrict C, std::size_t ldc) {
    T acc[MR][NR] = {};

    for (std::size_t k = 0; k < K; ++k) {
        T b[NR];
        for (std::size_t j = 0; j < NR; ++j) {
            b[j] = B[k * ldb + j];
        }
        for (std::size_t i = 0; i < MR; ++i) {
            const T a = A[i * lda + k];
            for (std::size_t j = 0; j < NR; ++j) {
                acc[i][j] += a * b[j];
            }
        }
    }

    for (std::size_t i = 0; i < MR; ++i) {
        for (std::size_t j = 0; j < NR; ++j) {
            C[i * ldc + j] += acc[i][j];
        }
    }
}
namespace details {

template <typename T, std::size_t MR, std::size_t NR>
inline void packed_kernel(std::size_t K, const T* __restrict Ap, const T* __restrict Bp,
                          T* __restrict C, std::size_t ldc, std::size_t mr, std::size_t nr) {
    T acc[MR][NR] = {};

    for (std::size_t k = 0; k < K; ++k) {
        const T* a = Ap + k * MR;
        const T* b = Bp + k * NR;
        for (std::size_t i = 0; i < MR; ++i) {
            for (std::size_t j = 0; j < NR; ++j) {
                acc[i][j] += a[i] * b[j];
            }
        }
    }

    if (mr == MR && nr == NR) {
        for (std::size_t i = 0; i < MR; ++i)
            for (std::size_t j = 0; j < NR; ++j)
                C[i * ldc + j] += acc[i][j];
    } else {
        for (std::size_t i = 0; i < mr; ++i)
            for (std::size_t j = 0; j < nr; ++j)
                C[i * ldc + j] += acc[i][j];
    }
}

template <typename T, std::size_t MR>
void pack_A(std::size_t mc, std::size_t kc, const T* A, std::size_t lda, T* Ap) {
    for (std::size_t ir = 0; ir < mc; ir += MR) {
        const std::size_t mr = std::min(MR, mc - ir);
        for (std::size_t k = 0; k < kc; ++k)
            for (std::size_t i = 0; i < MR; ++i)
                *Ap++ = (i < mr) ? A[(ir + i) * lda + k] : T{};
    }
}

template <typename T, std::size_t NR>
void pack_B(std::size_t kc, std::size_t nc, const T* B, std::size_t ldb, T* Bp) {
    for (std::size_t jr = 0; jr < nc; jr += NR) {
        const std::size_t nr = std::min(NR, nc - jr);
        for (std::size_t k = 0; k < kc; ++k)
            for (std::size_t j = 0; j < NR; ++j)
                *Bp++ = (j < nr) ? B[k * ldb + jr + j] : T{};
    }
}

constexpr std::size_t round_up(std::size_t x, std::size_t m) { return (x + m - 1) / m * m; }



template <typename T, std::size_t MR = 4, std::size_t NR = 8,
          std::size_t MC = 64, std::size_t KC = 256, std::size_t NC = 2048>
void gemm(std::size_t N, std::size_t M, std::size_t P,
          const T* __restrict A, const T* __restrict B, T* __restrict C) {
    std::vector<T> Ap(details::round_up(MC, MR) * KC);
    std::vector<T> Bp(KC * details::round_up(NC, NR));

    for (std::size_t jc = 0; jc < P; jc += NC) {
        const std::size_t nc = std::min(NC, P - jc);
        for (std::size_t pc = 0; pc < M; pc += KC) {
            const std::size_t kc = std::min(KC, M - pc);
            pack_B<T, NR>(kc, nc, B + pc * P + jc, P, Bp.data());

            for (std::size_t ic = 0; ic < N; ic += MC) {
                const std::size_t mc = std::min(MC, N - ic);
                pack_A<T, MR>(mc, kc, A + ic * M + pc, M, Ap.data());

                for (std::size_t jr = 0; jr < nc; jr += NR) {
                    const std::size_t nr = std::min(NR, nc - jr);
                    const T* bp = Bp.data() + (jr / NR) * kc * NR;
                    for (std::size_t ir = 0; ir < mc; ir += MR) {
                        const std::size_t mr = std::min(MR, mc - ir);
                        const T* ap = Ap.data() + (ir / MR) * kc * MR;
                        packed_kernel<T, MR, NR>(kc, ap, bp,
                                                 C + (ic + ir) * P + jc + jr, P, mr, nr);
                    }
                }
            }
        }
    }
}

} // namespace details

template <std::floating_point T, std::size_t N>
struct lu_result {
    square_matrix<T, N> lu;
    std::array<std::size_t, N> perm;
    T sign;
    bool singular;
};

template <std::floating_point T, std::size_t N>
[[nodiscard]] constexpr lu_result<T, N> lu_decompose(square_matrix<T, N> a) {
    std::array<std::size_t, N> perm{};
    for (std::size_t i = 0; i < N; ++i) perm[i] = i;
    T sign{1};

    for (std::size_t c = 0; c < N; ++c) {
        std::size_t pivot = c;
        for (std::size_t r = c + 1; r < N; ++r)
            if (std::abs(a[r, c]) > std::abs(a[pivot, c])) pivot = r;

        if (a[pivot, c] == T{0}) return {a, perm, sign, true};

        if (pivot != c) {
            for (std::size_t k = 0; k < N; ++k) std::swap(a[c, k], a[pivot, k]);
            std::swap(perm[c], perm[pivot]);
            sign = -sign;
        }

        for (std::size_t r = c + 1; r < N; ++r) {
            const T factor = a[r, c] / a[c, c];
            a[r, c] = factor;
            for (std::size_t k = c + 1; k < N; ++k)
                a[r, k] -= factor * a[c, k];
        }
    }

    return {a, perm, sign, false};
}

template <Numeric T, std::size_t N, std::size_t M, std::size_t P>
[[nodiscard]] constexpr matrix<T, N, P> matmul_microkernel(const matrix<T, N, M>& m1,
                                               const matrix<T, M, P>& m2) {
    matrix<T, N, P> res{};

    if consteval {
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t k = 0; k < M; ++k)
                for (std::size_t j = 0; j < P; ++j)
                    res[i, j] += m1[i, k] * m2[k, j];
    } else {
        details::gemm<T>(N, M, P, m1.raw().data(), m2.raw().data(), res.raw().data());
    }

    return res;
}

template <std::floating_point T, std::size_t N>
[[nodiscard]] constexpr T determinant(const square_matrix<T, N>& m) {
    if constexpr (N == 1) {
      return m[0, 0];
    } else if constexpr (N == 2) {
      return m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0];
    } else if constexpr (N == 3) {
      return m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
           - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
           + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]);
    } else if constexpr (N == 4) {
      const T s0 = m[2, 0] * m[3, 1] - m[2, 1] * m[3, 0];
      const T s1 = m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0];
      const T s2 = m[2, 0] * m[3, 3] - m[2, 3] * m[3, 0];
      const T s3 = m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1];
      const T s4 = m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1];
      const T s5 = m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2];

      return m[0, 0] * (m[1, 1] * s5 - m[1, 2] * s4 + m[1, 3] * s3)
           - m[0, 1] * (m[1, 0] * s5 - m[1, 2] * s2 + m[1, 3] * s1)
           + m[0, 2] * (m[1, 0] * s4 - m[1, 1] * s2 + m[1, 3] * s0)
           - m[0, 3] * (m[1, 0] * s3 - m[1, 1] * s1 + m[1, 2] * s0);
    } else {
      auto a = m;

      const auto x = lu_decompose(m);
      if(x.singular) return T{0};
      T det = x.sign;
      for(size_t i{}; i < N; ++i) {
        det *= x.lu[i, i];
      }
      return det;
    }
}
