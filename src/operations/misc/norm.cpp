#include "hicma/operations/misc.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"

#include <cstdint>
#include <cstdlib>


namespace hicma
{

float norm(const Matrix& A) { return norm_omm(A); }

define_method(float, norm_omm, (const Dense& A)) {
  float l2 = 0;
  timing::start("Norm(Dense)");
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      l2 += A(i, j) * A(i, j);
    }
  }
  timing::stop("Norm(Dense)");
  return l2;
}

define_method(float, norm_omm, (const LowRank& A)) { return norm(Dense(A)); }

define_method(float, norm_omm, (const Hierarchical& A)) {
  float l2 = 0;
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      l2 += norm(A(i, j));
    }
  }
  return l2;
}

define_method(float, norm_omm, (const Matrix& A)) {
  omm_error_handler("norm", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
