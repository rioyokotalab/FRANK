#include "FRANK/operations/misc.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/util/omm_error_handler.h"
#include "FRANK/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <cstdlib>


namespace FRANK
{

declare_method(double, norm_omm, (virtual_<const Matrix&>))

double norm(const Matrix& A) { return norm_omm(A); }

define_method(double, norm_omm, (const Dense& A)) {
  double l2 = 0;
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      l2 += A(i, j) * A(i, j);
    }
  }
  return l2;
}

define_method(double, norm_omm, (const LowRank& A)) { return norm(Dense(A)); }

define_method(double, norm_omm, (const Hierarchical& A)) {
  double l2 = 0;
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      l2 += norm(A(i, j));
    }
  }
  return l2;
}

define_method(double, norm_omm, (const Matrix& A)) {
  omm_error_handler("norm", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
