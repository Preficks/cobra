#pragma once

#include <concepts>
#include <cstddef>

inline constexpr size_t kspacetime_dim{4uz};

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;
