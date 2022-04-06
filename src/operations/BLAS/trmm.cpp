#include "hicma/operations/BLAS.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/operations/BLAS.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>


namespace hicma
{

void trmm(
  const Matrix& A, Matrix& B,
  const Side side, const Mode uplo, const char& trans, const char& diag,
  double alpha
) {
  trmm_omm(A, B, side, uplo, trans, diag, alpha);
}

void trmm(
  const Matrix& A, Matrix& B,
  const Side side, const Mode uplo,
  double alpha
) {
  trmm_omm(A, B, side, uplo, 'n', 'n', alpha);
}

define_method(
  void, trmm_omm,
  (
    const Dense& A, Dense& B,
    const Side side, const Mode uplo, const char& trans, const char& diag,
    double alpha
  )
) {
  // D D
  assert(A.dim[0] == A.dim[1]);
  assert(A.dim[0] == (side == Side::Left ? B.dim[0] : B.dim[1]));
  cblas_dtrmm(
    CblasRowMajor,
    side == Side::Left ? CblasLeft : CblasRight,
    uplo == Mode::Upper ? CblasUpper : CblasLower,
    trans == 't' ? CblasTrans : CblasNoTrans,
    diag == 'u' ? CblasUnit : CblasNonUnit,
    B.dim[0], B.dim[1], alpha, &A, A.stride, &B, B.stride
  );
}

define_method(
  void, trmm_omm,
  (
    const Dense& A, LowRank& B,
    const Side side, const Mode uplo,  const char& trans, const char& diag,
    double alpha
  )
) {
  // D LR
  assert(A.dim[0] == A.dim[1]);
  assert(A.dim[0] == (side == Side::Left ? B.dim[0] : B.dim[1]));
  if(side == Side::Left)
    trmm(A, B.U, side, uplo, trans, diag, alpha);
  else if(side == Side::Right)
    trmm(A, B.V, side, uplo, trans, diag, alpha);
}

define_method(
  void, trmm_omm,
  (
    const Hierarchical& A, Hierarchical& B,
    const Side side, const Mode uplo, const char& trans, const char& diag,
    double alpha
  )
) {
  // H H
  assert(A.dim[0] == A.dim[1]);
  assert(A.dim[0] == (side == Side::Left ? B.dim[0] : B.dim[1]));
  assert(trans != 't'); //TODO implement for transposed case: need transposed gemm complete
  Hierarchical B_copy(B);
  if(uplo == Mode::Upper) {
    if(side == Side::Left) {
      for(int64_t i=0; i<B.dim[0]; i++) {
        for(int64_t j=0; j<B.dim[1]; j++) {
          for(int64_t k=i; k<A.dim[1]; k++) {
            if(k == i)
              trmm(A(i, k), B(k, j), side, uplo, trans, diag, alpha);
            else
              gemm(A(i, k), B_copy(k, j), B(i, j), alpha, 1.);
          }
        }
      }
    }
    else if(side == Side::Right) {
      for(int64_t i=0; i<B.dim[0]; i++) {
        for(int64_t j=0; j<B.dim[1]; j++) {
          for(int64_t k=j; k>=0; k--) {
            if(k == j)
              trmm(A(k, j), B(i, k), side, uplo, trans, diag, alpha);
            else
              gemm(B_copy(i, k), A(k, j), B(i, j), alpha, 1);
          }
        }
      }
    }
  }
  else if(uplo == Mode::Lower) {
    if(side == Side::Left) {
      for(int64_t i=0; i<B.dim[0]; i++) {
        for(int64_t j=0; j<B.dim[1]; j++) {
          for(int64_t k=i; k>=0; k--) {
            if(k == i)
              trmm(A(i, k), B(k, j), side, uplo, trans, diag, alpha);
            else
              gemm(A(i, k), B_copy(k, j), B(i, j), alpha, 1);
          }
        }
      }
    }
    else if(side == Side::Right) {
      for(int64_t i=0; i<B.dim[0]; i++) {
        for(int64_t j=0; j<B.dim[1]; j++) {
          for(int64_t k=j; k<B.dim[1]; k++) {
            if(k == j)
              trmm(A(k, j), B(i, k), side, uplo, trans, diag, alpha);
            else
              gemm(B_copy(i, k), A(k, j), B(i, j), alpha, 1);
          }
        }
      }
    }
  }
}

// Fallback default, abort with error message
define_method(
  void, trmm_omm,
  (
    const Matrix& A, Matrix& B,
    const Side, const Mode, const char&, const char&, double
  )
) {
  omm_error_handler("trmm", {A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
