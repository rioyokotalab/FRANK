#include "hicma/operations/arithmetic.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/pre_scheduler.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>


namespace hicma
{

MatrixProxy operator-(const Matrix& A, const Matrix& B) {
  assert(get_n_rows(A) == get_n_rows(B));
  assert(get_n_cols(A) == get_n_cols(B));
  return subtraction_omm(A, B);
}

define_method(MatrixProxy, subtraction_omm, (const Dense& A, const Dense& B)) {
  Dense out(A);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      out(i, j) = A(i, j) - B(i, j);
    }
  }
  return out;
}

define_method(
  MatrixProxy, subtraction_omm, (const Matrix& A, const Matrix& B)
) {
  omm_error_handler("operator-", {A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
