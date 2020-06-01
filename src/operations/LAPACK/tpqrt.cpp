#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/operations/BLAS.h"
#include "hicma/util/omm_error_handler.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>


namespace hicma
{

void tpqrt(Matrix& A, Matrix& B, Matrix& T) { tpqrt_omm(A, B, T); }

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
  Dense BV_copy(B.V);
  B.V = gemm(B.S, B.V);
  B.S = 0.0;
  for(int64_t i = 0; i < std::min(B.S.dim[0], B.S.dim[1]); i++) {
    B.S(i, i) = 1.0;
  }
  tpqrt(A, B.V, T);
}

define_method(
  void, tpqrt_omm,
  (Hierarchical& A, Hierarchical& B, Hierarchical& T)
) {
  for(int64_t i = 0; i < B.dim[0]; i++) {
    for(int64_t j = 0; j < B.dim[1]; j++) {
      tpqrt(A(j, j), B(i, j), T(i, j));
      for(int64_t k = j+1; k < B.dim[1]; k++) {
        tpmqrt(B(i, j), T(i, j), A(j, k), B(i, k), true);
      }
    }
  }
}

// Fallback default, abort with error message
define_method(void, tpqrt_omm, (Matrix& A, Matrix& B, Matrix& T)) {
  omm_error_handler("tpqrt", {A, B, T}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
