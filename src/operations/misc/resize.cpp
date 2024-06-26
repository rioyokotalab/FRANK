#include "FRANK/operations/misc.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/classes/matrix_proxy.h"
#include "FRANK/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <utility>


namespace FRANK
{

declare_method(
  MatrixProxy, resize_omm, (virtual_<const Matrix&>, const int64_t, const int64_t)
)

MatrixProxy resize(const Matrix& A, const int64_t n_rows, const int64_t n_cols) {
  return resize_omm(A, n_rows, n_cols);
}

define_method(
  MatrixProxy, resize_omm, (const Dense& A, const int64_t n_rows, const int64_t n_cols)
) {
  assert(n_rows <= A.dim[0]);
  assert(n_cols <= A.dim[1]);
  Dense resized(n_rows, n_cols);
  A.copy_to(resized);
  return std::move(resized);
}

define_method(MatrixProxy, resize_omm, (const Matrix& A, const int64_t, const int64_t)) {
  omm_error_handler("resize", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
