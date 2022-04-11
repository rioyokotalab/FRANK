#include "hicma/operations/BLAS.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

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

void trsm(const Matrix& A, Matrix& B, const Mode uplo, const Side side) {
  assert(uplo == Mode::Upper || uplo == Mode::Lower);
  assert(side == Side::Left || side == Side::Right);
  trsm_omm(A, B, uplo, side);
}

define_method(
  void, trsm_omm,
  (const Hierarchical& A, Hierarchical& B, const Mode uplo, const Side side)
) {
  switch (uplo) {
    case Mode::Upper:
      switch (side) {
        case Side::Left:
          if (B.dim[1] == 1) {
            for (int64_t i=B.dim[0]-1; i>=0; i--) {
              for (int64_t j=B.dim[0]-1; j>i; j--) {
                gemm(A(i,j), B[j], B[i], -1, 1);
              }
              trsm(A(i,i), B[i], Mode::Upper, Side::Left);
            }
          } else {
            omm_error_handler(
                "Left upper with B.dim[1] != 1 trsm", {A, B}, __FILE__, __LINE__);
            std::abort();
          }
          break;
        case Side::Right:
          for (int64_t i=0; i<B.dim[0]; i++) {
            for (int64_t j=0; j<B.dim[1]; j++) {
              for (int64_t k=0; k<j; k++) {
                gemm(B(i,k), A(k,j), B(i,j), -1, 1);
              }
              trsm(A(j,j), B(i,j), Mode::Upper, Side::Right);
            }
          }
      }
      break;
    case Mode::Lower:
      switch (side) {
        case Side::Left:
          for (int64_t j=0; j<B.dim[1]; j++) {
            for (int64_t i=0; i<B.dim[0]; i++) {
              for (int64_t k=0; k<i; k++) {
                gemm(A(i,k), B(k,j), B(i,j), -1, 1);
              }
              trsm(A(i,i), B(i,j), Mode::Lower, Side::Left);
            }
          }
          break;
        case Side::Right:
          omm_error_handler("Right lower trsm", {A, B}, __FILE__, __LINE__);
          std::abort();
      }
      break;
  }
}

define_method(void, trsm_omm, (const Dense& A, Dense& B, const Mode uplo, const Side side)) {
  cblas_dtrsm(
    CblasRowMajor,
    side==Side::Left?CblasLeft:CblasRight,
    uplo==Mode::Upper?CblasUpper:CblasLower,
    CblasNoTrans,
    uplo==Mode::Upper?CblasNonUnit:CblasUnit,
    B.dim[0], B.dim[1],
    1,
    &A, A.stride,
    &B, B.stride
  );
}

define_method(void, trsm_omm, (const Matrix& A, LowRank& B, const Mode uplo, const Side side)) {
  switch (side) {
  case Side::Left:
    trsm(A, B.U, uplo, side);
    break;
  case Side::Right:
    trsm(A, B.V, uplo, side);
    break;
  }
}

define_method(
  void, trsm_omm,
  (const Hierarchical& A, Dense& B, const Mode uplo, const Side side)
) {
  Hierarchical BH = split(
    B, side==Side::Left?A.dim[0]:1, side==Side::Left?1:A.dim[1]
  );
  trsm(A, BH, uplo, side);
}

// Fallback default, abort with error message
define_method(void, trsm_omm, (const Matrix& A, Matrix& B, const Mode, const Side)) {
  omm_error_handler("trsm", {A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
