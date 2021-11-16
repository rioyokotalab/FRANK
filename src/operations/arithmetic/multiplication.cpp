#include "hicma/operations/arithmetic.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/pre_scheduler.h"

#include <cstdint>
#include <cstdlib>


namespace hicma
{

Matrix& operator*=(Matrix& A, double b) {
  return multiplication_omm(A, b);
}

define_method(
  Matrix&, multiplication_omm, (Dense& A, double b)
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) *= b;
    }
  }
  return A;
}

define_method(
  Matrix&, multiplication_omm, (LowRank& A, double b)
) {
  A.S *= b;
  return A;
}

define_method(
  Matrix&, multiplication_omm, (Hierarchical& A, double b)
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) *= b;
    }
  }
  return A;
}

define_method(Matrix&, multiplication_omm, (Matrix& A, double)) {
  omm_error_handler("operator*<double>", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
