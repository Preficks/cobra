#pragma once

#include "config.hpp"
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <array>
#include <ostream>
#include <sys/types.h>
#include <iostream>
#include <utility>

template <Numeric T, size_t... Dims>
class tensor {
  static constexpr size_t rank          = sizeof...(Dims);
  static constexpr size_t size         = (1 * ... * Dims);
  static constexpr std::array<size_t, rank> dims{Dims...};

  std::array<T, size> data;

  public:

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

  constexpr tensor& operator +=(const tensor& rhs) {
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::plus<>{});
    return *this;
  }

  constexpr tensor& operator -=(const tensor& rhs) {
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::minus<>{});
    return *this;
  }




};

template<size_t Skip, size_t R>
constexpr std::array<size_t, R - 1> drop(std::array<size_t, R>& in) {
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
