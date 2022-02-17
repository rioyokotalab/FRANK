#include "hicma/classes/initialization_helpers/matrix_kernels/arange_kernel.h"

#include "hicma/classes/dense.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

// explicit template initialization (these are the only available types)
template class ArangeKernel<float>;
template class ArangeKernel<double>;

template<typename U>
std::unique_ptr<MatrixKernel<U>> ArangeKernel<U>::clone() const {
  return std::make_unique<ArangeKernel<U>>(*this);
}

declare_method(void, apply_arange_kernel, (virtual_<Matrix&>, int64_t, int64_t))

template<typename U>
void ArangeKernel<U>::apply(Matrix& A, int64_t row_start, int64_t col_start) const {
  apply_arange_kernel(A, row_start, col_start);
}

template<typename T>
void apply_dense(Dense<T>& A, int64_t row_start, int64_t col_start) {
  for(int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      A(i, j) = (row_start+i) * A.stride + col_start + j;
    }
  }
}

define_method(void, apply_arange_kernel, (Dense<float>& A, int64_t row_start, int64_t col_start)) {
  apply_dense(A, row_start, col_start);
}

define_method(void, apply_arange_kernel, (Dense<double>& A, int64_t row_start, int64_t col_start)) {
  apply_dense(A, row_start, col_start);
}

define_method(void, apply_arange_kernel, (Matrix& A, int64_t, int64_t)) {
  omm_error_handler("apply_arange_kernel", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
