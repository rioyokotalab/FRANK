#include "FRANK/operations/LAPACK.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/classes/matrix_proxy.h"
#include "FRANK/functions.h"
#include "FRANK/operations/BLAS.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"
#include "FRANK/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>


namespace FRANK
{

declare_method(
  void, rq_omm,
  (virtual_<Matrix&>, virtual_<Matrix&>, virtual_<Matrix&>)
)

void rq(Matrix& A, Matrix& R, Matrix& Q) { rq_omm(A, R, Q); }

define_method(void, rq_omm, (Dense& A, Dense& R, Dense& Q)) {
  assert(R.dim[0] == A.dim[0]);
  assert(R.dim[1] == Q.dim[0]);
  assert(Q.dim[1] == A.dim[1]);
  const int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<double> tau(k);
  LAPACKE_dgerqf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
  // Copy upper triangular into R
  for(int64_t i=0; i<R.dim[0]; i++) {
    for(int64_t j=std::max(i+A.dim[1]-A.dim[0], (int64_t)0); j<A.dim[1]; j++) {
      R(i, j+R.dim[1]-A.dim[1]) = A(i, j);
    }
  }
  // Copy strictly lower part into Q
  for(int64_t i=std::max(A.dim[0]-A.dim[1], (int64_t)0); i<A.dim[0]; i++) {
    for(int64_t j=0; j<(i+A.dim[1]-A.dim[0]); j++) {
      Q(i+Q.dim[0]-A.dim[0], j) = A(i, j);
    }
  }
  // TODO Consider making special function for this. Performance heavy and not
  // always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense derivative that remains in elementary reflector
  // form, uses dormrq instead of gemm and can be transformed to Dense via
  // dorgrq!
  LAPACKE_dorgrq(
    LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]
  );
}

} // namespace FRANK
