#include "hicma/operations/arithmetic.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/node.h"
#include "hicma/util/omm_error_handler.h"

#include <cstdint>


namespace hicma
{

Node& operator*=(Node& A, double b) {
  return multiplication_omm(A, b);
}

define_method(
  Node&, multiplication_omm, (Dense& A, double b)
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) *= b;
    }
  }
  return A;
}

define_method(
  Node&, multiplication_omm, (LowRank& A, double b)
) {
  A.S() *= b;
  return A;
}

define_method(
  Node&, multiplication_omm, (Hierarchical& A, double b)
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) *= b;
    }
  }
  return A;
}

define_method(Node&, multiplication_omm, (Node& A, [[maybe_unused]] double b)) {
  omm_error_handler("operator*<double>", {A}, __FILE__, __LINE__);
  abort();
}

} // namespace hicma
