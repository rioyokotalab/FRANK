#include "FRANK/operations/arithmetic.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/classes/matrix_proxy.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>


namespace FRANK
{

declare_method(
  MatrixProxy, subtraction_omm,
  (virtual_<const Matrix&>, virtual_<const Matrix&>)
)

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

} // namespace FRANK
