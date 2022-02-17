#include "hicma/classes/initialization_helpers/matrix_kernels/helmholtznd_kernel.h"

#include "hicma/classes/dense.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cmath>
#include <utility>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class HelmholtzndKernel<float>;
template class HelmholtzndKernel<double>;

template<typename U>
HelmholtzndKernel<U>::HelmholtzndKernel(const vec2d<U>& params) :
                      ParameterizedKernel<U>(params) {}

template<typename U>
HelmholtzndKernel<U>::HelmholtzndKernel(vec2d<U>&& params) :
                      ParameterizedKernel<U>(std::move(params)) {}

template<typename U>
std::unique_ptr<MatrixKernel<U>> HelmholtzndKernel<U>::clone() const {
  return std::make_unique<HelmholtzndKernel<U>>(*this);
}

declare_method(void, apply_helmholtznd_kernel, (virtual_<Matrix&>, const vec2d<double>&, int64_t, int64_t))

declare_method(void, apply_helmholtznd_kernel, (virtual_<Matrix&>, const vec2d<float>&, int64_t, int64_t))

template<typename U>
void HelmholtzndKernel<U>::apply(Matrix& A, int64_t row_start, int64_t col_start) const {
  apply_helmholtznd_kernel(A, this->params, row_start, col_start);
}

template<typename T, typename U>
void apply_dense(Dense<T>& A, const vec2d<U> params, int64_t row_start, int64_t col_start) {
  for(int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      U rij = 0;
      for (size_t k=0; k<params.size(); ++k) {
        rij += (params[k][row_start+i] - params[k][col_start+j])
               * (params[k][row_start+i] - params[k][col_start+j]);
      }
      // relies on implicit conversion
      A(i, j) = std::exp(-1.0 * rij) / (std::sqrt(rij) + 1e-3);
    }
  }
}

define_method(void, apply_helmholtznd_kernel, (Dense<float>& A, const vec2d<float>& params, int64_t row_start, int64_t col_start)) {
  apply_dense(A, params, row_start, col_start);
}

define_method(void, apply_helmholtznd_kernel, (Dense<float>& A, const vec2d<double>& params, int64_t row_start, int64_t col_start)) {
  apply_dense(A, params, row_start, col_start);
}

define_method(void, apply_helmholtznd_kernel, (Dense<double>& A, const vec2d<float>& params, int64_t row_start, int64_t col_start)) {
  apply_dense(A, params, row_start, col_start);
}

define_method(void, apply_helmholtznd_kernel, (Dense<double>& A, const vec2d<double>& params, int64_t row_start, int64_t col_start)) {
  apply_dense(A, params, row_start, col_start);
}

define_method(void, apply_helmholtznd_kernel, (Matrix& A, const vec2d<double>&, int64_t, int64_t)) {
  omm_error_handler("apply_helmholtznd_kernel", {A}, __FILE__, __LINE__);
  std::abort();
}

define_method(void, apply_helmholtznd_kernel, (Matrix& A, const vec2d<float>&, int64_t, int64_t)) {
  omm_error_handler("apply_helmholtznd_kernel", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
