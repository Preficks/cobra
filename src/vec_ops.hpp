#include "tensor.hpp"
#include <algorithm>
#include <functional>

template <Numeric T>
[[nodiscard]] constexpr vec3<T> cross(const vec3<T>& v1, const vec3<T>& v2) {
  return { v1[1] * v2[2] - v1[2] * v2[1],
           v1[0] * v2[2] - v1[2] * v2[0],
           v1[0] * v2[1] - v1[1] * v2[0]};
}

template <Numeric T>
[[nodiscard]] constexpr T det(const matrix<T, 3, 3> m) {
  return  m[0, 0] * m[1, 1] * m[2, 2]
        + m[0, 1] * m[1, 2] * m[2, 3]
        + m[0, 2] * m[1, 0] * m[2, 1]
        - m[2, 0] * m[1, 1] * m[0, 1]
        - m[2, 1] * m[1, 2] * m[0, 0]
        - m[2, 2] * m[1, 0] * m[0, 1];
}

template <Numeric T>
[[nodiscard]] constexpr T det(const matrix<T, 2, 2> m) {
  return  m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0];
}

template <Numeric T, std::size_t N>
constexpr tensor<T, N>& operator *=(tensor<T, N>& lhs, const tensor<T, N>& rhs) {
    std::transform(lhs.raw().begin(), lhs.raw().end(), rhs.raw().begin(), lhs.raw().begin(), std::multiplies<>{});
    return lhs;
}

template <Numeric T, std::size_t N>
constexpr tensor<T, N>& operator /=(tensor<T, N>& lhs, const tensor<T, N>& rhs) {
    std::transform(lhs.raw().begin(), lhs.raw().end(), rhs.raw().begin(), lhs.raw().begin(), std::divides<>{});
    return lhs;
}

template <Numeric T, std::size_t N>
constexpr tensor<T, N>& operator *=(tensor<T, N>& lhs, const T rhs) {
    std::transform(lhs.raw().begin(), lhs.raw().end(), lhs.raw().begin(), [rhs](T a) {
      return a * rhs;
    });
    return lhs;
}

template <Numeric T, std::size_t N>
constexpr tensor<T, N>& operator /=(tensor<T, N>& lhs, const T rhs) {
    std::transform(lhs.raw().begin(), lhs.raw().end(), lhs.raw().begin(), [rhs](T a) {
      return a / rhs;
    });
    return lhs;
}

template <Numeric T, std::size_t N>
constexpr tensor<T, N>& operator +=(tensor<T, N>& lhs, const tensor<T, N>& rhs) {
    std::transform(lhs.raw().begin(), lhs.raw().end(), rhs.raw().begin(), lhs.raw().begin(), std::plus<>{});
    return lhs;
}

template <Numeric T, std::size_t N>
constexpr tensor<T, N>& operator -=(tensor<T, N>& lhs, const tensor<T, N>& rhs) {
    std::transform(lhs.raw().begin(), lhs.raw().end(), rhs.raw().begin(), lhs.raw().begin(), std::minus<>{});
    return lhs;
}

template <Numeric T, std::size_t N>
[[nodiscard]] constexpr tensor<T, N> operator +(tensor<T, N> lhs, const tensor<T, N>& rhs) {
    lhs += rhs;
    return lhs;
}

template <Numeric T, std::size_t N>
[[nodiscard]] constexpr tensor<T, N> operator -(tensor<T, N> lhs, const tensor<T, N>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <Numeric T, std::size_t N>
[[nodiscard]] constexpr tensor<T, N> operator *(tensor<T, N> lhs, const tensor<T, N>& rhs) {
    lhs *= rhs;
    return lhs;
}

template <Numeric T, std::size_t N>
[[nodiscard]] constexpr tensor<T, N> operator /(tensor<T, N> lhs, const tensor<T, N>& rhs) {
    lhs /= rhs;
    return lhs;
}

template <Numeric T, std::size_t N>
[[nodiscard]] constexpr T dot(tensor<T, N> a, tensor<T,N> b) {
  return contract<0, 0>(a, b);
}

template <Numeric T, std::size_t N>
[[nodiscard]] constexpr T norm2(tensor<T, N> a) {
  T res{};
  for(T x : a.data) {
    res += x * x;
  }

  return sqrt(res);
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
    std::vector<T> Ap(round_up(MC, MR) * KC);
    std::vector<T> Bp(KC * round_up(NC, NR));

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
        gemm<T>(N, M, P, m1.raw().data(), m2.raw().data(), res.raw().data());
    }

    return res;
}
