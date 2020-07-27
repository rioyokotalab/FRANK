#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/initialization_helpers/basis_tracker.h"
#include "hicma/operations/BLAS.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

std::tuple<MatrixProxy, MatrixProxy> getrf(Matrix& A) {
  clear_trackers();
  return getrf_omm(A);
}

define_method(MatrixPair, getrf_omm, (Hierarchical& A)) {
  Hierarchical L(A.dim[0], A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; i++) {
    std::tie(L(i, i), A(i, i)) = getrf_omm(A(i, i));
    for (int64_t i_c=i+1; i_c<L.dim[0]; i_c++) {
      L(i_c, i) = std::move(A(i_c, i));
      trsm(A(i, i), L(i_c, i), TRSM_UPPER, TRSM_RIGHT);
    }
    for (int64_t j=i+1; j<A.dim[1]; j++) {
      trsm(L(i, i), A(i, j), TRSM_LOWER, TRSM_LEFT);
    }
    for (int64_t i_c=i+1; i_c<L.dim[0]; i_c++) {
      for (int64_t k=i+1; k<A.dim[1]; k++) {
        gemm(L(i_c, i), A(i, k), A(i_c, k), -1, 1);
      }
    }
  }
  return {std::move(L), std::move(A)};
}

define_method(MatrixPair, getrf_omm, (Dense& A)) {
  timing::start("DGETRF");
  std::vector<int> ipiv(std::min(A.dim[0], A.dim[1]));
  LAPACKE_dgetrf(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &ipiv[0]
  );
  Dense L(A.dim[0], A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<i; j++) {
      L(i, j) = A(i, j);
      A(i, j) = 0;
    }
    L(i, i) = 1;
  }
  timing::stop("DGETRF");
  return {std::move(L), std::move(A)};
}

define_method(MatrixPair, getrf_omm, (Matrix& A)) {
  omm_error_handler("getrf", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
