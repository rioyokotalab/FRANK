#include "FRANK/operations/LAPACK.h"

#include "FRANK/definitions.h"
#include "FRANK/classes/dense.h"
#include "FRANK/classes/empty.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/operations/BLAS.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"
#include "FRANK/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>


namespace FRANK
{

declare_method(
  MatrixPair, getrf_omm,
  (virtual_<Matrix&>)
)

std::tuple<MatrixProxy, MatrixProxy> getrf(Matrix& A) {
  std::tuple<MatrixProxy, MatrixProxy> out = getrf_omm(A);
  return out;
}

define_method(MatrixPair, getrf_omm, (Hierarchical& A)) {
  Hierarchical L(A.dim[0], A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; i++) {
    std::tie(L(i, i), A(i, i)) = getrf_omm(A(i, i));
    for (int64_t i_c=i+1; i_c<L.dim[0]; i_c++) {
      L(i_c, i) = std::move(A(i_c, i));
      A(i_c, i) = Empty(get_n_rows(L(i_c, i)), get_n_cols(L(i_c, i)));
      trsm(A(i, i), L(i_c, i), Mode::Upper, Side::Right);
    }
    for (int64_t j=i+1; j<A.dim[1]; j++) {
      L(i, j) = Empty(get_n_rows(A(i, j)), get_n_cols(A(i, j)));
      trsm(L(i, i), A(i, j), Mode::Lower, Side::Left);
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
  Dense L(A.dim[0], A.dim[1]);
  std::vector<int> ipiv(std::min(A.dim[0], A.dim[1]));
  LAPACKE_dgetrf(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &ipiv[0]
  );
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<i; j++) {
      L(i, j) = A(i, j);
      A(i, j) = 0;
    }
    L(i, i) = 1;
  }
  return {std::move(L), std::move(A)};
}

define_method(MatrixPair, getrf_omm, (Matrix& A)) {
  omm_error_handler("getrf", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
