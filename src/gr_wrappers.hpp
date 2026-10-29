#include "tensor.hpp"
#include <variant>
#include "config.hpp"

namespace gr {

enum class Variance { Co, Contra };

template<size_t N, Variance V>
struct Index {
  static constexpr size_t dim = N;
  static constexpr Variance variance = V;
};

template <size_t N = gr::kspacetime_dim> using Up = Index<N, Variance::Contra>;
template <size_t N = gr::kspacetime_dim> using Down = Index<N, Variance::Co>;

template <int... Labels> struct Symmetry {};
template <typename... Is> struct index_list {};

template <typename T, typename Indices, typename Sym = /* no symmetry for this rank */>
class Tensor;

template <typename T, typename... Is, int... Labels>
class Tensor<T, index_list<Is...>, Symmetry<Labels...>> {
  static_assert(sizeof...(Labels) == sizeof...(Is), "one symmetry label per index");
  static constexpr size_t rank = sizeof...(Is);
  static constexpr size_t dims[] = { Is::dim... };
  static constexpr Variance variances[] = { Is::variance... };

  // TODO: TENSOR CONTRACTION

};

using metric = Tensor<double, index_list<Down<>, Down<>>, Symmetry<1, 1>>;
using christoffel = Tensor<double, index_list<Up<>, Down<>, Down<>>, Symmetry<1, 2, 2>>;

}
