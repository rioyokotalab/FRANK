#include "hicma/classes/initialization_helpers/matrix_kernels/cauchy2d_kernel.h"

#include "hicma/classes/dense.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <utility>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class Cauchy2dKernel<float>;
template class Cauchy2dKernel<double>;

template<typename U>
Cauchy2dKernel<U>::Cauchy2dKernel(const vec2d<U>& params) :
                   ParameterizedKernel<U>(params) {}

template<typename U>
Cauchy2dKernel<U>::Cauchy2dKernel(vec2d<U>&& params) :
                   ParameterizedKernel<U>(std::move(params)) {}

template<typename U>
std::unique_ptr<MatrixKernel<U>> Cauchy2dKernel<U>::clone() const {
  return std::make_unique<Cauchy2dKernel<U>>(*this);
}

template<typename U>
std::unique_ptr<MatrixKernel<U>> Cauchy2dKernel<U>::move_clone() {
  return std::make_unique<Cauchy2dKernel<U>>(std::move(*this));
}

declare_method(void, apply_cauchy2d_kernel, (virtual_<Matrix&>, const vec2d<double>&, int64_t, int64_t))

declare_method(void, apply_cauchy2d_kernel, (virtual_<Matrix&>, const vec2d<float>&, int64_t, int64_t))

template<typename U>
void Cauchy2dKernel<U>::apply(Matrix& A, int64_t row_start, int64_t col_start) const {
  apply_cauchy2d_kernel(A, this->params, row_start, col_start);
}

template<typename T, typename U>
void apply_dense(Dense<T>& A, const vec2d<U> params, int64_t row_start, int64_t col_start) {
  assert(params.size()>1);
  assert(params[0].size()>=A.dim[0]);
  assert(params[1].size()>=A.dim[1]);

  for(int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      U rij = (params[0][row_start+i] - params[1][col_start+j]) + 1e-2;
      // relies on implicit conversion
      A(i, j) = 1 / rij;
    }
  }
}

define_method(void, apply_cauchy2d_kernel, (Dense<float>& A, const vec2d<float>& params, int64_t row_start, int64_t col_start)) {
  apply_dense(A, params, row_start, col_start);
}

define_method(void, apply_cauchy2d_kernel, (Dense<float>& A, const vec2d<double>& params, int64_t row_start, int64_t col_start)) {
  apply_dense(A, params, row_start, col_start);
}

define_method(void, apply_cauchy2d_kernel, (Dense<double>& A, const vec2d<float>& params, int64_t row_start, int64_t col_start)) {
  apply_dense(A, params, row_start, col_start);
}

define_method(void, apply_cauchy2d_kernel, (Dense<double>& A, const vec2d<double>& params, int64_t row_start, int64_t col_start)) {
  apply_dense(A, params, row_start, col_start);
}

define_method(void, apply_cauchy2d_kernel, (Matrix& A, const vec2d<double>&, int64_t, int64_t)) {
  omm_error_handler("apply_cauchy2d_kernel", {A}, __FILE__, __LINE__);
  std::abort();
}

define_method(void, apply_cauchy2d_kernel, (Matrix& A, const vec2d<float>&, int64_t, int64_t)) {
  omm_error_handler("apply_cauchy2d_kernel", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
