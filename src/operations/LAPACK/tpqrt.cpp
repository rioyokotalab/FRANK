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

// single precision
define_method(void, tpqrt_omm, (Dense<float>& A, Dense<float>& B, Dense<float>& T)) {
  LAPACKE_stpqrt2(
    LAPACK_ROW_MAJOR,
    B.dim[0], B.dim[1], 0,
    &A, A.stride,
    &B, B.stride,
    &T, T.stride
  );
}

// double precision
define_method(void, tpqrt_omm, (Dense<double>& A, Dense<double>& B, Dense<double>& T)) {
  LAPACKE_dtpqrt2(
    LAPACK_ROW_MAJOR,
    B.dim[0], B.dim[1], 0,
    &A, A.stride,
    &B, B.stride,
    &T, T.stride
  );
}

template<typename T>
void tpqrt_d_lr_d(Dense<T>& A, LowRank<T>& B, Dense<T>& S) {
  Dense<T> BV_copy(B.V);
  B.V = gemm(B.S, BV_copy);
  B.S = 0.0;
  for(int64_t i = 0; i < std::min(B.S.dim[0], B.S.dim[1]); i++) {
    B.S(i, i) = 1.0;
  }
  tpqrt(A, B.V, S);
}

define_method(void, tpqrt_omm, (Dense<float>& A, LowRank<float>& B, Dense<float>& T)) {
  tpqrt_d_lr_d(A, B, T);
}

define_method(void, tpqrt_omm, (Dense<double>& A, LowRank<double>& B, Dense<double>& T)) {
  tpqrt_d_lr_d(A, B, T);
}

template<typename T>
void tpqrt_h_h_h(Hierarchical<T>& A, Hierarchical<T>& B, Hierarchical<T>& S) {
  for(int64_t i = 0; i < B.dim[0]; i++) {
    for(int64_t j = 0; j < B.dim[1]; j++) {
      tpqrt(A(j, j), B(i, j), S(i, j));
      for(int64_t k = j+1; k < B.dim[1]; k++) {
        tpmqrt(B(i, j), S(i, j), A(j, k), B(i, k), true);
      }
    }
  }
}

define_method(
  void, tpqrt_omm,
  (Hierarchical<float>& A, Hierarchical<float>& B, Hierarchical<float>& T)
) {
  tpqrt_h_h_h(A, B, T);
}

define_method(
  void, tpqrt_omm,
  (Hierarchical<double>& A, Hierarchical<double>& B, Hierarchical<double>& T)
) {
  tpqrt_h_h_h(A, B, T);
}

// Fallback default, abort with error message
define_method(void, tpqrt_omm, (Matrix& A, Matrix& B, Matrix& T)) {
  omm_error_handler("tpqrt", {A, B, T}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
