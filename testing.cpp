#include <sys/types.h>
#include <iostream>
#include "src/rank1_tensor.hpp"
#include "src/rank2_tensor.hpp"



int main() {
  rank2_tensor<double, 2, 2> a{{{1.0, 0.0}, {0.0, 1.0}}};
  rank2_tensor<double, 2, 2> b{{{1,0}, {0, 1}}};
  std::cout << a * b << '\n';
}
