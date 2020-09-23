#include "hicma/operations/misc.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/pre_scheduler.h"

#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <cstdint>


namespace hicma
{

MatrixProxy resize(const Matrix& A, int64_t n_rows, int64_t n_cols) {
  return resize_omm(A, n_rows, n_cols);
}

define_method(
  MatrixProxy, resize_omm, (const Dense& A, int64_t n_rows, int64_t n_cols)
) {
  assert(n_rows <= A.dim[0]);
  assert(n_cols <= A.dim[1]);
  Dense resized(n_rows, n_cols);
  add_copy_task(A, resized);
  return resized;
}

define_method(MatrixProxy, resize_omm, (const Matrix& A, int64_t, int64_t)) {
  omm_error_handler("resize", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma