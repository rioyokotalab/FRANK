#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

void rq(Matrix& A, Matrix& R, Matrix& Q) { rq_omm(A, R, Q); }

define_method(void, rq_omm, (Dense& A, Dense& R, Dense& Q)) {
  assert(R.dim[0] == A.dim[0]);
  assert(R.dim[1] == Q.dim[0]);
  assert(Q.dim[1] == A.dim[1]);
  int64_t k = std::min(A.dim[0], A.dim[1]);
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

} // namespace hicma
