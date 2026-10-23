#include "config.hpp"
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <array>
#include <ostream>
#include <sys/types.h>
#include <iostream>

template <Numeric T, std::size_t N = kspacetime_dim>
class rank1_tensor {
  std::array<T, N> data;
  public:

  constexpr rank1_tensor<T, N>(std::initializer_list<T> l) {
    for(int i{}; i < N; ++i) {
      data[i] = *(l.begin() + i);
    }
  }

  constexpr size_t size() {
    return data.size();
  }

  [[nodiscard]] constexpr T mag2() {
    T res;
    for(const T& entry : data) { res += entry * entry; }

    return res;
  }

  [[nodiscard]] constexpr T mag() const {
    T res{};
    for(const T& entry : data) { res += entry * entry; }

    return std::sqrt(res);
  }

  [[nodiscard]] constexpr rank1_tensor<T, N> normalized() const
    requires std::floating_point<T> {
      auto m = mag();

      if (m == 0) {
          return *this;
      }

      rank1_tensor<T, N> result = *this;
      result /= m;
      return result;
  }
  constexpr rank1_tensor<T, N>& operator +=(const rank1_tensor<T, N>& rhs) {
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::plus<>{});

    return *this;
  }

  constexpr rank1_tensor<T, N>& operator -=(const rank1_tensor<T, N>& rhs) {
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), [](int a, int b) {
      return a - b;
    });

    return *this;
  }

  [[nodiscard]] friend rank1_tensor<T, N> operator -(rank1_tensor<T, N> lhs, const rank1_tensor<T, N>& rhs) {
    lhs -= rhs;
    return lhs;
  }

  [[nodiscard]] friend rank1_tensor<T, N> operator +(rank1_tensor<T, N> lhs, const rank1_tensor<T, N>& rhs) {
    lhs += rhs;
    return lhs;
  }

  // Element Wise
  constexpr rank1_tensor<T, N>& operator *=(const rank1_tensor<T>& rhs) {
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), [](T a, T b){
        return a * b;
      });

    return *this;
  }

  [[nodiscard]] friend constexpr rank1_tensor<T, N> operator *(rank1_tensor<T, N> lhs, const rank1_tensor<T, N>& rhs) {
    lhs *= rhs;
    return lhs;
  }

  constexpr rank1_tensor<T, N>& operator *=(const Numeric auto a) {
    std::transform(data.begin(), data.end(), data.begin(), [a](T b) {
      return a * b;
    });

    return *this;
  }

  [[nodiscard]] friend constexpr rank1_tensor<T, N> operator *(const Numeric auto lhs, rank1_tensor<T, N> rhs) {
    rhs *= lhs;
    return rhs;
  }

  [[nodiscard]] friend constexpr rank1_tensor<T, N> operator *(rank1_tensor<T, N> lhs, const Numeric auto rhs) {
    lhs *= rhs;
    return lhs;
  }

  constexpr rank1_tensor<T, N>& operator /=(const rank1_tensor<T>& rhs) {
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), [](T a, T b){
        return a / b;
      });

    return *this;
  }

  [[nodiscard]] friend constexpr rank1_tensor<T, N> operator /(rank1_tensor<T, N> lhs, const rank1_tensor<T, N>& rhs) {
    lhs /= rhs;
    return lhs;
  }

  constexpr rank1_tensor<T, N>& operator /=(const Numeric auto b) {
    std::transform(data.begin(), data.end(), data.begin(), [b](T a) {
      return a / b;
    });

    return *this;
  }

  [[nodiscard]] friend constexpr rank1_tensor<T, N> operator /(const Numeric auto lhs, rank1_tensor<T, N> rhs) {
    rhs /= lhs;
    return rhs;
  }

  [[nodiscard]] friend constexpr rank1_tensor<T, N> operator /(rank1_tensor<T, N> lhs, const Numeric auto rhs) {
    lhs /= rhs;
    return lhs;
  }

  constexpr T operator [](int i) const {
    return data[i];
  }

  friend constexpr std::ostream& operator <<(std::ostream& os, const rank1_tensor<T, N>& t) {
    os << "[";
    for(int i{}; i < N; i++) {
      if(i != N - 1) {
        os << t.data[i] << ", ";
      } else {
        os << t.data[i] << "]";
      }
    }

    return os;
  }
};
