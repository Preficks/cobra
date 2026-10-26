#include "tensor.hpp"
#include <algorithm>
#include <functional>

template <Numeric T>
[[nodiscard]] constexpr vec3<T> cross(const vec3<T>& v1, const vec3<T>& v2) {
  return { v1[1] * v2[2] - v1[2] * v2[1],
           v1[0] * v2[2] - v1[2] * v2[0],
           v1[0] * v2[1] - v1[1] * v2[0]};
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
