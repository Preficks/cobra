#pragma once

#include "../config.hpp"
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <array>
#include <ostream>
#include <sys/types.h>
#include <iostream>


template<Numeric T, size_t N = kspacetime_dim, size_t M = kspacetime_dim>
class rank2_tensor {
  std::array<T, N * M> data;

  public:

  rank2_tensor<T, N, M>(std::initializer_list<T> il){
    std::ranges::copy(il, data.begin());
  }

  rank2_tensor(const T (&init)[N][M]) : data{} {
      for (std::size_t i = 0; i < N; ++i) {
          for (std::size_t j = 0; j < M; ++j) {
              data[i * M + j] = init[i][j];
          }
      }
  }

  constexpr rank2_tensor<T, N, M>& operator +=(const rank2_tensor<T, N, M>& rhs) {
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::plus<>{});

    return *this;
  }

  constexpr rank2_tensor<T, N, M>& operator -=(const rank2_tensor<T, N, M>& rhs) {
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), [](T a, T b){
      return a - b;
    });

    return *this;
  }

  constexpr rank2_tensor<T, N, M>& operator *=(const rank2_tensor<T, N, M>& rhs) {
    static_assert(N == M, "In place multiplication requires square matrices");

    *this = *this * rhs;

    return *this;
  }

  template <size_t P>
  [[nodiscard]] friend constexpr rank2_tensor<T, N, P> operator *(const rank2_tensor<T, N, M>& lhs,const rank2_tensor<T, M, P>& rhs) {
    rank2_tensor<T, N, P> res{};

    for(int i{}; i < N; ++i) {
      for(int j{}; j < P; ++j) {
        T sum{};
        for(int k{}; k < M; ++k) {
          sum += lhs[i, k] * rhs[k, j];
        }

        res[i ,j] = sum;
      }
    }

    return res;
  }

  [[nodiscard]] constexpr T& operator [](size_t i, size_t j) {
    return data[(i * M + j)];
  }

  [[nodiscard]] constexpr const T& operator[](size_t i, size_t j) const {
    return data[(i * M + j)];
  }

  constexpr friend std::ostream& operator <<(std::ostream& os, const rank2_tensor<T, N, M>& t) {
    for(int i{}; i < N; ++i) {
      os << "[";
      for(int j{}; j < M; ++j) {
        os << t.data[i * M + j];
        if(j != M - 1) {
          os << ", ";
        }
      }
      os << "]\n";
    }

    return os;
  }
};
