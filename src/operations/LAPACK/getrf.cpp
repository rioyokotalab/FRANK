#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/intitialization_helpers/basis_tracker.h"
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

std::tuple<MatrixProxy, MatrixProxy> getrf(Matrix& A) { return getrf_omm(A); }

declare_method(
  void, decuple_col_basis, (virtual_<Matrix&>, BasisTracker<BasisKey>&)
)

define_method(
  void, decuple_col_basis, (LowRank& A, BasisTracker<BasisKey>& tracker)
) {
  timing::start("decoupling");
  if (!tracker.has_basis(A.U)) {
    tracker[A.U] = A.U;
  }
  A.U = share_basis(tracker[A.U]);
  timing::stop("decoupling");
}

define_method(void, decuple_col_basis, (Matrix&, BasisTracker<BasisKey>&)) {
  // Do nothing
}

declare_method(
  void, decuple_row_basis, (virtual_<Matrix&>, BasisTracker<BasisKey>&)
)

define_method(
  void, decuple_row_basis, (LowRank& A, BasisTracker<BasisKey>& tracker)
) {
  timing::start("decoupling");
  if (!tracker.has_basis(A.V)) {
    tracker[A.V] = A.V;
  }
  A.V = share_basis(tracker[A.V]);
  timing::stop("decoupling");
}

define_method(void, decuple_row_basis, (Matrix&, BasisTracker<BasisKey>&)) {
  // Do nothing
}

define_method(MatrixPair, getrf_omm, (Hierarchical& A)) {
  Hierarchical L(A.dim[0], A.dim[1]);
  // TODO This will only work for matrices with a single layer! The basis
  // tracker would need to be shared from outside the functions...
  BasisTracker<BasisKey> basis_tracker;
  for (int64_t i=0; i<A.dim[0]; i++) {
    std::tie(L(i, i), A(i, i)) = getrf(A(i,i));
    BasisTracker<BasisKey> trsm_tracker;
    for (int64_t i_c=i+1; i_c<L.dim[0]; i_c++) {
      L(i_c, i) = std::move(A(i_c, i));
      decuple_row_basis(L(i_c, i), basis_tracker);
      trsm_omm(A(i, i), L(i_c, i), TRSM_UPPER, TRSM_RIGHT, trsm_tracker);
    }
    for (int64_t j=i+1; j<A.dim[1]; j++) {
      trsm_omm(L(i, i), A(i, j), TRSM_LOWER, TRSM_LEFT, trsm_tracker);
    }
    for (int64_t i_c=i+1; i_c<L.dim[0]; i_c++) {
      for (int64_t k=i+1; k<A.dim[1]; k++) {
        gemm(L(i_c,i), A(i,k), A(i_c,k), -1, 1);
      }
    }
    // Decouple column basis of lower part after operations so that bases are
    // shared during the gemm call (faster LR+=LR)
    for (int64_t i_c=i+1; i_c<L.dim[0]; i_c++) {
      decuple_col_basis(L(i_c, i), basis_tracker);
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
