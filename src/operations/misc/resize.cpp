#include "hicma/operations/misc.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <cstdint>


namespace hicma
{

MatrixProxy resize(const Matrix& A, int64_t n_rows, int64_t n_cols) {
  return resize_omm(A, n_rows, n_cols);
}

template<typename T>
Dense<T> dense_resize(const Dense<T>& A, int64_t n_rows, int64_t n_cols) {
  assert(n_rows <= A.dim[0]);
  assert(n_cols <= A.dim[1]);
  Dense<T> resized(n_rows, n_cols);
  A.copy_to(resized);
  return resized;
}

define_method(
  MatrixProxy, resize_omm, (const Dense<double>& A, int64_t n_rows, int64_t n_cols)
) {
  return dense_resize(A, n_rows, n_cols);
}

define_method(
  MatrixProxy, resize_omm, (const Dense<float>& A, int64_t n_rows, int64_t n_cols)
) {
  return dense_resize(A, n_rows, n_cols);
}

define_method(MatrixProxy, resize_omm, (const Matrix& A, int64_t, int64_t)) {
  omm_error_handler("resize", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
