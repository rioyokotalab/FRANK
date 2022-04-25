#include "FRANK/operations/BLAS.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/operations/BLAS.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"
#include "FRANK/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>


namespace FRANK
{

declare_method(
  void, trmm_omm,
  (
    virtual_<const Matrix&>, virtual_<Matrix&>,
    const Side, const Mode, const char&, const char&, const double
  )
)

void trmm(
  const Matrix& A, Matrix& B,
  const Side side, const Mode uplo, const char& trans, const char& diag,
  const double alpha
) {
  trmm_omm(A, B, side, uplo, trans, diag, alpha);
}

void trmm(
  const Matrix& A, Matrix& B,
  const Side side, const Mode uplo,
  const double alpha
) {
  trmm_omm(A, B, side, uplo, 'n', 'n', alpha);
}

define_method(
  void, trmm_omm,
  (
    const Dense& A, Dense& B,
    const Side side, const Mode uplo, const char& trans, const char& diag,
    const double alpha
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
    const double alpha
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
    const Dense& A, Hierarchical& B,
    const Side side, const Mode uplo,  const char& trans, const char& diag,
    const double alpha
  )
) {
  // D H
  assert(A.dim[0] == A.dim[1]);
  assert(A.dim[0] == (side == Side::Left ? get_n_rows(B) : get_n_cols(B)));
  const Hierarchical AH = split(A,
                                side == Side::Left ? B.dim[0] : B.dim[1],
                                side == Side::Left ? B.dim[0] : B.dim[1],
                                false);
  trmm(AH, B, side, uplo, trans, diag, alpha);
}

define_method(
  void, trmm_omm,
  (
    const Hierarchical& A, Dense& B,
    const Side side, const Mode uplo,  const char& trans, const char& diag,
    const double alpha
  )
) {
  // H D
  assert(A.dim[0] == A.dim[1]);
  assert(get_n_rows(A) == (side == Side::Left ? B.dim[0] : B.dim[1]));
  Hierarchical BH = split(B,
                          side == Side::Left ? A.dim[1] : 1,
                          side == Side::Left ? 1 : A.dim[0],
                          false);
  trmm(A, BH, side, uplo, trans, diag, alpha);
}

define_method(
  void, trmm_omm,
  (
    const Hierarchical& A, LowRank& B,
    const Side side, const Mode uplo,  const char& trans, const char& diag,
    const double alpha
  )
) {
  // H LR
  assert(A.dim[0] == A.dim[1]);
  assert(get_n_rows(A) == (side == Side::Left ? B.dim[0] : B.dim[1]));
  Hierarchical BH = split(
      side == Side::Left ? B.U : B.V,
      side == Side::Left ? A.dim[1] : 1,
      side == Side::Left ? 1 : A.dim[0],
      false);
  trmm(A, BH, side, uplo, trans, diag, alpha);
}

define_method(
  void, trmm_omm,
  (
    const Hierarchical& A, Hierarchical& B,
    const Side side, const Mode uplo, const char& trans, const char& diag,
    const double alpha
  )
) {
  // H H
  assert(A.dim[0] == A.dim[1]);
  assert(A.dim[0] == (side == Side::Left ? B.dim[0] : B.dim[1]));
  assert(trans != 't'); //TODO implement for transposed case: need transposed gemm complete
  const Hierarchical B_copy(B);
  switch(uplo) {
    case Mode::Upper:
      switch(side) {
        case Side::Left:
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
          break;
        case Side::Right:
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
          break;
      }
      break;
    case Mode::Lower:
      switch(side) {
        case Side::Left:
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
          break;
        case Side::Right:
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
          break;
      }
      break;
  }
}

// Fallback default, abort with error message
define_method(
  void, trmm_omm,
  (
    const Matrix& A, Matrix& B,
    const Side, const Mode, const char&, const char&, const double
  )
) {
  omm_error_handler("trmm", {A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
