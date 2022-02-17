#include "hicma/classes/initialization_helpers/matrix_kernels/random_normal_kernel.h"

#include "hicma/classes/dense.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <utility>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class RandomNormalKernel<float>;
template class RandomNormalKernel<double>;

/* initialize static members */
template<typename U>
std::normal_distribution<U> RandomNormalKernel<U>::dist = std::normal_distribution<U>(0.0, 1.0);

template<typename U>
std::mt19937 RandomNormalKernel<U>::gen = std::mt19937();

template<typename U>
std::random_device RandomNormalKernel<U>::rd = std::random_device();

/* member functions */
template<typename U>
RandomNormalKernel<U>::RandomNormalKernel(uint32_t seed) : seed(seed) {}

template<typename U>
std::unique_ptr<MatrixKernel<U>> RandomNormalKernel<U>::clone() const {
  return std::make_unique<RandomNormalKernel<U>>(*this);
}

template<typename U>
std::unique_ptr<MatrixKernel<U>> RandomNormalKernel<U>::move_clone() {
  return std::make_unique<RandomNormalKernel<U>>(std::move(*this));
}

declare_method(void, apply_random_normal_kernel, (virtual_<Matrix&>, std::mt19937&, std::normal_distribution<float>&))

declare_method(void, apply_random_normal_kernel, (virtual_<Matrix&>, std::mt19937&, std::normal_distribution<double>&))

template<typename U>
void RandomNormalKernel<U>::apply(Matrix& A, int64_t row_start, int64_t col_start) const {
  apply(A, true, row_start, col_start);
}

template<typename U>
void RandomNormalKernel<U>::apply(Matrix& A,  bool deterministic_seed, int64_t, int64_t) const {
  if (deterministic_seed) {
    gen.seed(seed);
  }
  else {
    gen.seed(rd());
  }
  apply_random_normal_kernel(A, gen, dist);
}

template<typename T, typename U>
void apply_dense(Dense<T>& A, std::mt19937& gen, std::normal_distribution<U>& dist) {
  for(int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      A(i, j) = dist(gen);
    }
  }
}

define_method(void, apply_random_normal_kernel, (Dense<float>& A, std::mt19937& gen, std::normal_distribution<float>& dist)) {
  apply_dense(A, gen, dist);
}

define_method(void, apply_random_normal_kernel, (Dense<float>& A, std::mt19937& gen, std::normal_distribution<double>& dist)) {
  apply_dense(A, gen, dist);
}

define_method(void, apply_random_normal_kernel, (Dense<double>& A, std::mt19937& gen, std::normal_distribution<float>& dist)) {
  apply_dense(A, gen, dist);
}

define_method(void, apply_random_normal_kernel, (Dense<double>& A, std::mt19937& gen, std::normal_distribution<double>& dist)) {
  apply_dense(A, gen, dist);
}

define_method(void, apply_random_normal_kernel, (Matrix& A, std::mt19937&, std::normal_distribution<float>&)) {
  omm_error_handler("apply_random_normal_kernel", {A}, __FILE__, __LINE__);
  std::abort();
}

define_method(void, apply_random_normal_kernel, (Matrix& A, std::mt19937&, std::normal_distribution<double>&)) {
  omm_error_handler("apply_random_normal_kernel", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
