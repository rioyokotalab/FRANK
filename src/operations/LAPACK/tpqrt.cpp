#include "hicma/operations/LAPACK/tpqrt.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/LAPACK/tpmqrt.h"

#include <algorithm>
#include <iostream>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

void tpqrt(
  Node& A, Node& B, Node& T
) {
  tpqrt_omm(A, B, T);
}

BEGIN_SPECIALIZATION(
  tpqrt_omm, void,
  Dense& A, Dense& B, Dense& T
) {
  LAPACKE_dtpqrt2(
    LAPACK_ROW_MAJOR,
    B.dim[0], B.dim[1], 0,
    &A[0], A.dim[1],
    &B[0], B.dim[1],
    &T[0], T.dim[1]
  );
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpqrt_omm, void,
  Dense& A, LowRank& B, Dense& T
) {
  Dense BV_copy(B.V);
  gemm(B.S, BV_copy, B.V, 1, 0);
  B.S = 0.0;
  for(int i=0; i<std::min(B.S.dim[0], B.S.dim[1]); i++) B.S(i, i) = 1.0;
  tpqrt(A, B.V, T);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpqrt_omm, void,
  Hierarchical& A, Hierarchical& B, Hierarchical& T
) {
  for(int i = 0; i < B.dim[0]; i++) {
    for(int j = 0; j < B.dim[1]; j++) {
      tpqrt(A(j, j), B(i, j), T(i, j));
      for(int k = j+1; k < B.dim[1]; k++) {
        tpmqrt(B(i, j), T(i, j), A(j, k), B(i, k), true);
      }
    }
  }
} END_SPECIALIZATION;

// Fallback default, abort with error message
BEGIN_SPECIALIZATION(
  tpqrt_omm, void,
  Node& A, Node& B, Node& T
) {
  std::cerr << "tpqrt(";
  std::cerr << A.type() << "," << B.type() << "," << T.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
