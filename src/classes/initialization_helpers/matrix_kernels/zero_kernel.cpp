#include "hicma/classes/initialization_helpers/matrix_kernels/zero_kernel.h"

#include "hicma/classes/dense.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

// explicit template initialization (these are the only available types)
template class ZeroKernel<float>;
template class ZeroKernel<double>;

template<typename U>
std::unique_ptr<MatrixKernel<U>> ZeroKernel<U>::clone() const {
  return std::make_unique<ZeroKernel<U>>(*this);
}

declare_method(void, apply_zero_kernel, (virtual_<Matrix&>))

template<typename U>
void ZeroKernel<U>::apply(Matrix& A, int64_t, int64_t) const {
  apply_zero_kernel(A);
}

template<typename T>
void apply_dense(Dense<T>& A) {
  for(int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      A(i, j) = 0;
    }
  }
}

define_method(void, apply_zero_kernel, (Dense<float>& A)) {
  apply_dense(A);
}

define_method(void, apply_zero_kernel, (Dense<double>& A)) {
  apply_dense(A);
}

define_method(void, apply_zero_kernel, (Matrix& A)) {
  omm_error_handler("apply_zero_kernel", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
