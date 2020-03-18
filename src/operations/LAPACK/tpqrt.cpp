#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS.h"

#include <algorithm>
#include <iostream>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"


namespace hicma
{

void tpqrt(
  Node& A, Node& B, Node& T
) {
  tpqrt_omm(A, B, T);
}

define_method(void, tpqrt_omm, (Dense& A, Dense& B, Dense& T)) {
  LAPACKE_dtpqrt2(
    LAPACK_ROW_MAJOR,
    B.dim[0], B.dim[1], 0,
    &A, A.stride,
    &B, B.stride,
    &T, T.stride
  );
}

define_method(void, tpqrt_omm, (Dense& A, LowRank& B, Dense& T)) {
  Dense BV_copy(B.V());
  gemm(B.S(), BV_copy, B.V(), 1, 0);
  B.S() = 0.0;
  for(int i=0; i<std::min(B.S().dim[0], B.S().dim[1]); i++) B.S()(i, i) = 1.0;
  tpqrt(A, B.V(), T);
}

define_method(
  void, tpqrt_omm,
  (Hierarchical& A, Hierarchical& B, Hierarchical& T)
) {
  for(int i = 0; i < B.dim[0]; i++) {
    for(int j = 0; j < B.dim[1]; j++) {
      tpqrt(A(j, j), B(i, j), T(i, j));
      for(int k = j+1; k < B.dim[1]; k++) {
        tpmqrt(B(i, j), T(i, j), A(j, k), B(i, k), true);
      }
    }
  }
}

// Fallback default, abort with error message
define_method(void, tpqrt_omm, (Node& A, Node& B, Node& T)) {
  std::cerr << "tpqrt(";
  std::cerr << A.type() << "," << B.type() << "," << T.type();
  std::cerr << ") undefined." << std::endl;
  abort();
}

} // namespace hicma
