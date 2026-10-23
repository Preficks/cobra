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
