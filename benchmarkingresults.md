g++ -std=c++23 -O2 -march=native testing.cpp -o testing

# Matmul specific
100 x 100 matrices

Elapsed time:
  677771 ns
  677 µs
  0.677771 ms

# Standard contraction
100x100 matrices

checksum: 124.886
Elapsed time:
  695376 ns
  695 µs
  0.695376 ms

# After optimizations (ikj order) for standard contraction: 
500 x 500 matrices: 

Elapsed time:
  24037450 ns
  24037 µs
  24.0374 ms

# Matmul specific (micro kernels, ikj order, memory blocking)
500 x 500 matrices:

Elapsed time:
  249578 ns
  249 µs
  7.249578 ms
