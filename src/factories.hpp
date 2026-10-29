#include "tensor.hpp"
#include "config.hpp"
#include <algorithm>
#include <random>

template <Numeric T, size_t... Dims>
constexpr tensor<T, Dims...> full(T value) {
  tensor<T, Dims...> res{};

  std::ranges::fill(res, value);
  return res;
}

template<Numeric T, size_t... Dims>
constexpr tensor<T, Dims...> filled(T value) { return full<T, Dims...>(value); }

template <Numeric T, size_t... Dims>
constexpr tensor<T, Dims...> ones() { return full<T, Dims...>(T{1}); }



template <Numeric T, size_t N>
constexpr square_matrix<T, N> identity(T value) {
  square_matrix<T, N> res{};
  for(size_t i{}; i < N; i++) {
    res[i, i] = T{1};
  }
  return res;
}

template <Numeric T, std::size_t... Dims>
[[nodiscard]] tensor<T, Dims...> random(std::uint32_t seed, T min = T{-1}, T max = T{1}) {
    std::mt19937 rng{seed};
    tensor<T, Dims...> res{};

    if constexpr (std::floating_point<T>) {
        std::uniform_real_distribution<T> dist{min, max};
        std::ranges::generate(res, [&] { return dist(rng); });
    } else {
        std::uniform_int_distribution<T> dist{min, max};
        std::ranges::generate(res, [&] { return dist(rng); });
    }

    return res;
}
