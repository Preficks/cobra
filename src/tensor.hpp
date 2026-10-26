#pragma once

#include "config.hpp"
#include <concepts>
#include <cstddef>
#include <functional>
#include <array>
#include <ostream>
#include <sys/types.h>
#include <utility>

template <Numeric T, size_t... Dims>
class tensor {
  public:
  static constexpr size_t rank          = sizeof...(Dims);
  static constexpr size_t size         = (1 * ... * Dims);
  static constexpr std::array<size_t, rank> dims{Dims...};

  constexpr auto begin()       { return data.begin(); }
  constexpr auto end()         { return data.end(); }
  constexpr auto begin() const { return data.begin(); }
  constexpr auto end()   const { return data.end(); }

  private:
  std::array<T, size> data;

  public:

  constexpr auto& raw() { return data; }
  constexpr const auto& raw() const { return data; }

  constexpr tensor() = default;

  template <std::convertible_to<T>... Args>
    requires (sizeof...(Args) == (1 * ... * Dims))
  constexpr tensor(Args... args) : data{static_cast<T>(args)...} {}

  template <std::integral... Idx>
    requires(sizeof...(Idx) == rank)
  constexpr T& operator[](Idx... idx) {
    std::array<std::size_t, rank> i{static_cast<std::size_t>(idx)...};
    std::size_t flat = 0;
    for (std::size_t k = 0; k < rank; ++k) {
      flat = flat * dims[k] + i[k];
    }
    return data[flat];
  }

  template <std::integral... Idx>
    requires(sizeof...(Idx) == rank)
  constexpr const T& operator[](Idx... idx) const{
    std::array<std::size_t, rank> i{static_cast<std::size_t>(idx)...};
    std::size_t flat = 0;
    for (std::size_t k = 0; k < rank; ++k) {
      flat = flat * dims[k] + i[k];
    }
    return data[flat];
  }

  constexpr tensor& operator +=(const tensor& rhs) {
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::plus<>{});
    return *this;
  }

  constexpr tensor& operator -=(const tensor& rhs) {
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::minus<>{});
    return *this;
  }
};

template<Numeric T, size_t... Dims>
[[nodiscard]] constexpr bool allclose(const tensor<T, Dims...>& a, const tensor<T, Dims...>& b, T rtol = 1e-9, T atol = 1e-12) {
  for(size_t i{}; i < a.size; ++i) {
    const T diff = std::abs(a.data[i] - b.data[i]);
    if(diff > atol && diff > rtol * std::abs(a.data[i])) {
      return false;
    }
  }
  return true;
}

template<size_t Skip, size_t R>
constexpr std::array<size_t, R - 1> drop(const std::array<size_t, R>& in) {
  std::array<size_t, R - 1> res{};

  for(int i{}, o{}; i < R; ++i) {
    if(i != Skip) { res[o++] = in[i]; }
  }

  return res;
}

template<size_t R1, size_t R2>
constexpr std::array<size_t, R1 + R2> concat(std::array<size_t, R1> a1, std::array<size_t, R2> a2) {
  std::array<size_t, R1 + R2> res{};

  for(size_t i{}; i < R1; ++i) { res[i] = a1[i]; }
  for(size_t i{}; i < R2; ++i) { res[R1 + i] = a2[i]; }

  return res;
}

template <Numeric T, auto Shape, size_t... Is>
auto make_tensor_type(std::index_sequence<Is...>) -> tensor<T, Shape[Is]...>;

template <Numeric T, auto Shape>
using tensor_from_shape =
  decltype(make_tensor_type<T, Shape>(std::make_index_sequence<Shape.size()>{}));

template <std::size_t R>
constexpr std::size_t flatten(const std::array<std::size_t, R>& idx, const std::array<std::size_t, R>& dims) {
    std::size_t flat = 0;
    for (std::size_t k = 0; k < R; ++k) flat = flat * dims[k] + idx[k];
    return flat;
}

template <std::size_t R>
constexpr std::array<std::size_t, R> unflatten(std::size_t flat, const std::array<std::size_t, R>& dims) {
    std::array<std::size_t, R> idx{};
    for (std::size_t k = R; k-- > 0;) {
        idx[k] = flat % dims[k];
        flat /= dims[k];
    }
    return idx;
}

template <Numeric T, std::size_t... Dims>
std::ostream& operator<<(std::ostream& os, const tensor<T, Dims...>& t) {
    os << '[';
    for (std::size_t i = 0; i < t.raw().size(); ++i) {
        os << t.raw()[i];
        if (i + 1 < t.raw().size()) os << ", ";
    }
    return os << ']';
}

template <std::size_t I, std::size_t J, Numeric T, std::size_t... A, std::size_t... B>
  requires (I < sizeof...(A) && J < sizeof...(B))
[[nodiscard]] constexpr auto contract(const tensor<T, A...>& a, const tensor<T, B...>& b) {
  constexpr std::array<std::size_t, sizeof...(A)> adims{A...};
  constexpr std::array<std::size_t, sizeof...(B)> bdims{B...};
  static_assert(adims[I] == bdims[J], "contracted dimensions must match");

  constexpr auto rdims = concat(drop<I>(adims), drop<J>(bdims));
  using result_t = tensor_from_shape<T, rdims>;
  constexpr std::size_t a_free = sizeof...(A) - 1;

  result_t res{};
  for (std::size_t f = 0; f < result_t::size; ++f) {
    auto r = unflatten(f, rdims);
    std::array<std::size_t, sizeof...(A)> ai{};
    std::array<std::size_t, sizeof...(B)> bi{};
    for (std::size_t k = 0, o = 0; k < ai.size(); ++k)
      if (k != I) ai[k] = r[o++];
    for (std::size_t k = 0, o = a_free; k < bi.size(); ++k)
      if (k != J) bi[k] = r[o++];

    T sum{};
    for (std::size_t s = 0; s < adims[I]; ++s) {
      ai[I] = s;
      bi[J] = s;
      sum += a.raw()[flatten(ai, adims)] * b.raw()[flatten(bi, bdims)];
    }
    res.raw()[f] = sum;
  }
  return res;
}

template <std::size_t R>
constexpr std::size_t product(const std::array<std::size_t, R>& d, std::size_t from, std::size_t to) {
    std::size_t p = 1;
    for (std::size_t k = from; k < to; ++k) p *= d[k];
    return p;
}

template <std::size_t I, std::size_t J, Numeric T, std::size_t... A, std::size_t... B>
  requires (I < sizeof...(A) && J < sizeof...(B))
[[nodiscard]] constexpr auto contract1(const tensor<T, A...>& a, const tensor<T, B...>& b) {
  constexpr std::array<std::size_t, sizeof...(A)> adims{A...};
  constexpr std::array<std::size_t, sizeof...(B)> bdims{B...};
  static_assert(adims[I] == bdims[J], "contracted dimensions must match");

  constexpr auto rdims = concat(drop<I>(adims), drop<J>(bdims));
  using result_t = tensor_from_shape<T, rdims>;

  constexpr std::size_t n         = adims[I];
  constexpr std::size_t a_before  = product(adims, 0, I);
  constexpr std::size_t a_after   = product(adims, I + 1, adims.size());
  constexpr std::size_t b_before  = product(bdims, 0, J);
  constexpr std::size_t b_after   = product(bdims, J + 1, bdims.size());
  constexpr std::size_t b_free    = b_before * b_after;

  const auto& A_ = a.raw();
  const auto& B_ = b.raw();
  result_t res{};
  auto& R_ = res.raw();

  for (std::size_t ab = 0; ab < a_before; ++ab) {
    for (std::size_t aa = 0; aa < a_after; ++aa) {
      const std::size_t r_base = (ab * a_after + aa) * b_free;

      for (std::size_t s = 0; s < n; ++s) {
        const T x = A_[(ab * n + s) * a_after + aa];

        for (std::size_t bb = 0; bb < b_before; ++bb) {
          const std::size_t b_base = (bb * n + s) * b_after;
          const std::size_t r_row  = r_base + bb * b_after;

          for (std::size_t ba = 0; ba < b_after; ++ba) {
            R_[r_row + ba] += x * B_[b_base + ba];
          }
        }
      }
    }
  }

  return res;
}


template <Numeric T, size_t N>
using vector = tensor<T, N>;

template <Numeric T> using vec3 = tensor<T, 3>;
template <Numeric T> using vec4 = tensor<T, 4>;

template <Numeric T, size_t N, size_t M>
using matrix = tensor<T, N, M>;

template <Numeric T, size_t N>
using square_matrix = tensor<T, N, N>;
