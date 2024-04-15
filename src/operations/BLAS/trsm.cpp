#include "hicma/operations/BLAS.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"
#include "hicma/util/global_key_value.h"

#include "yorel/yomm2/cute.hpp"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <cassert>
#include <cstdint>
#include <cstdlib>


namespace hicma
{

void trsm(const Matrix& A, Matrix& B, int uplo, int lr) {
  assert(uplo == TRSM_UPPER || uplo == TRSM_LOWER);
  assert(lr == TRSM_LEFT || lr == TRSM_RIGHT);
  trsm_omm(A, B, uplo, lr);
}

template<typename T>
void hierarchical_trsm(const Hierarchical<T>& A, Hierarchical<T>& B, int uplo, int lr){
  switch (uplo) {
  case TRSM_UPPER:
    switch (lr) {
    case TRSM_LEFT:
      if (B.dim[1] == 1) {
        for (int64_t i=B.dim[0]-1; i>=0; i--) {
          for (int64_t j=B.dim[0]-1; j>i; j--) {
            gemm(A(i,j), B[j], B[i], -1, 1);
          }
          trsm(A(i,i), B[i], TRSM_UPPER, TRSM_LEFT);
        }
      } else {
        omm_error_handler(
          "Left upper with B.dim[1] != 1 trsm", {A, B}, __FILE__, __LINE__);
        std::abort();
      }
      break;
    case TRSM_RIGHT:
      for (int64_t i=0; i<B.dim[0]; i++) {
        for (int64_t j=0; j<B.dim[1]; j++) {
          for (int64_t k=0; k<j; k++) {
            gemm(B(i,k), A(k,j), B(i,j), -1, 1);
          }
          trsm(A(j,j), B(i,j), TRSM_UPPER, TRSM_RIGHT);
        }
      }
    }
    break;
  case TRSM_LOWER:
    switch (lr) {
    case TRSM_LEFT:
      for (int64_t j=0; j<B.dim[1]; j++) {
        for (int64_t i=0; i<B.dim[0]; i++) {
          for (int64_t k=0; k<i; k++) {
            gemm(A(i,k), B(k,j), B(i,j), -1, 1);
          }
          trsm(A(i,i), B(i,j), TRSM_LOWER, TRSM_LEFT);
        }
      }
      break;
    case TRSM_RIGHT:
      omm_error_handler("Right lower trsm", {A, B}, __FILE__, __LINE__);
      std::abort();
    }
    break;
  }
}

define_method(
  void, trsm_omm,
  (const Hierarchical<float>& A, Hierarchical<float>& B, int uplo, int lr)
) {
 hierarchical_trsm(A, B, uplo, lr);
}

define_method(
  void, trsm_omm,
  (const Hierarchical<double>& A, Hierarchical<double>& B, int uplo, int lr)
) {
 hierarchical_trsm(A, B, uplo, lr);
}

// single precision
define_method(void, trsm_omm, (const Dense<float>& A, Dense<float>& B, int uplo, int lr)) {
  //timing::start("STRSM");
  cblas_strsm(
    CblasRowMajor,
    lr==TRSM_LEFT?CblasLeft:CblasRight,
    uplo==TRSM_UPPER?CblasUpper:CblasLower,
    CblasNoTrans,
    uplo==TRSM_UPPER?CblasNonUnit:CblasUnit,
    B.dim[0], B.dim[1],
    1,
    &A, A.stride,
    &B, B.stride
  );
  add_trsm_flops(B.dim[0], B.dim[1], lr);
  //timing::stop("STRSM");
}

// double precision
define_method(void, trsm_omm, (const Dense<double>& A, Dense<double>& B, int uplo, int lr)) {
  //timing::start("DTRSM");
  cblas_dtrsm(
    CblasRowMajor,
    lr==TRSM_LEFT?CblasLeft:CblasRight,
    uplo==TRSM_UPPER?CblasUpper:CblasLower,
    CblasNoTrans,
    uplo==TRSM_UPPER?CblasNonUnit:CblasUnit,
    B.dim[0], B.dim[1],
    1,
    &A, A.stride,
    &B, B.stride
  );
  add_trsm_flops(B.dim[0], B.dim[1], lr);
  //timing::stop("DTRSM");
}

template<typename T>
void low_rank_trsm(const Matrix& A, LowRank<T>& B, int uplo, int lr) {
  switch (lr) {
  case TRSM_LEFT:
    trsm(A, B.U, uplo, lr);
    break;
  case TRSM_RIGHT:
    trsm(A, B.V, uplo, lr);
    break;
  }
}

define_method(void, trsm_omm, (const Matrix& A, LowRank<float>& B, int uplo, int lr)) {
  low_rank_trsm(A, B, uplo, lr);
}

define_method(void, trsm_omm, (const Matrix& A, LowRank<double>& B, int uplo, int lr)) {
  low_rank_trsm(A, B, uplo, lr);
}

template<typename T>
void hierarchical_dense_trsm(const Hierarchical<T>& A, Dense<T>& B, int uplo, int lr) {
  Hierarchical<T> BH = split<T>(
    B, lr==TRSM_LEFT?A.dim[0]:1, lr==TRSM_LEFT?1:A.dim[1]
  );
  trsm(A, BH, uplo, lr);
}

define_method(
  void, trsm_omm,
  (const Hierarchical<float>& A, Dense<float>& B, int uplo, int lr)
) {
  hierarchical_dense_trsm(A, B, uplo, lr);
}

define_method(
  void, trsm_omm,
  (const Hierarchical<double>& A, Dense<double>& B, int uplo, int lr)
) {
  hierarchical_dense_trsm(A, B, uplo, lr);
}

// Fallback default, abort with error message
define_method(void, trsm_omm, (const Matrix& A, Matrix& B, int, int)) {
  omm_error_handler("trsm", {A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
