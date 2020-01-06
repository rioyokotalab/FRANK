#include "hicma/operations/LAPACK/getrf.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/BLAS/trsm.h"
#include "hicma/util/timer.h"

#include <iostream>
#include <tuple>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

std::tuple<NodeProxy, NodeProxy> getrf(Node& A) {
  return getrf_omm(A);
}

BEGIN_SPECIALIZATION(getrf_omm, NodePair, Hierarchical& A) {
  Hierarchical L(A.dim[0], A.dim[1]);
  for (int i=0; i<A.dim[0]; i++) {
    std::tie(L(i, i), A(i, i)) = getrf(A(i,i));
    for (int i_c=i+1; i_c<L.dim[0]; i_c++) {
      L(i_c, i) = std::move(A(i_c, i));
      trsm(A(i,i), L(i_c,i), 'u');
    }
    for (int j=i+1; j<A.dim[1]; j++) {
      trsm(L(i,i), A(i,j), 'l');
    }
    for (int i_c=i+1; i_c<L.dim[0]; i_c++) {
      gemm(L(i_c,i), A(i,i_c), A(i_c,i_c), -1, 1);
      L(i_c, i_c) = A(i_c, i_c);
      for (int k=i+1; k<i_c; k++) {
        L(i_c, k) = std::move(A(i_c, k));
        gemm(L(i_c,i), A(i,k), L(i_c,k), -1, 1);
      }
    }
    for (int i_c=i+1; i_c<A.dim[0]; i_c++) {
      for (int k=i_c+1; k<A.dim[1]; k++) {
        gemm(L(i_c, i), A(i,k), A(i_c, k), -1, 1);
      }
    }
  }
  return {std::move(L), std::move(A)};
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(getrf_omm, NodePair, Dense& A) {
  start("-DGETRF");
  std::vector<int> ipiv(std::min(A.dim[0], A.dim[1]));
  LAPACKE_dgetrf(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A[0], A.dim[1],
    &ipiv[0]
  );
  Dense L(A.dim[0], A.dim[1]);
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<i; j++) {
      L(i, j) = A(i, j);
      A(i, j) = 0;
    }
    L(i, i) = 1;
  }
  stop("-DGETRF",false);
  return {std::move(L), std::move(A)};
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(getrf_omm, NodePair, Node& A) {
  std::cerr << "getrf(" << A.type() << ") undefined!" << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
