#include "FRANK/operations/arithmetic.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <cstdlib>


namespace FRANK
{

declare_method(
  Matrix&, multiplication_omm,
  (virtual_<Matrix&>, double)
)

Matrix& operator*=(Matrix& A, const double b) {
  return multiplication_omm(A, b);
}

define_method(
  Matrix&, multiplication_omm, (Dense& A, const double b)
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) *= b;
    }
  }
  return A;
}

define_method(
  Matrix&, multiplication_omm, (LowRank& A, const double b)
) {
  A.S *= b;
  return A;
}

define_method(
  Matrix&, multiplication_omm, (Hierarchical& A, const double b)
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) *= b;
    }
  }
  return A;
}

define_method(Matrix&, multiplication_omm, (Matrix& A, const double)) {
  omm_error_handler("operator*<double>", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
