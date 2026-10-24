#include <chrono>
#include <memory>
#include <sys/types.h>
#include <iostream>
#include "src/tensor.hpp"
#include "src/vec_ops.hpp"
#include <random>

int main() {
  const size_t trials = 100uz;
  const size_t size = 100uz;

  auto mat1 = std::make_unique<matrix<double, size, size>>();
  auto mat2 = std::make_unique<matrix<double, size, size>>();

  std::mt19937 rng{42};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  for (auto& x : mat1->raw()) x = dist(rng);
  for (auto& x : mat2->raw()) x = dist(rng);

  double checksum = 0.0;
  auto start_time = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < trials; ++i) {
      //auto res = contract1<1, 0>(*mat1, *mat2);
      auto res = matmul_microkernel(*mat1, *mat2);
      checksum += res.raw()[i % res.size];
  }
  auto end_time = std::chrono::steady_clock::now();

  std::cout << "checksum: " << checksum << '\n';

  auto elapsed_duration = (end_time - start_time) / trials;

  auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_duration).count();
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_duration).count();

  // Alternative: Convert directly to a floating-point representation of milliseconds
  std::chrono::duration<double, std::milli> milliseconds_fp = elapsed_duration;

  std::cout << "Elapsed time:\n";
  std::cout << "  " << nanoseconds << " ns\n";
  std::cout << "  " << microseconds << " µs\n";
  std::cout << "  " << milliseconds_fp.count() << " ms\n";

}
