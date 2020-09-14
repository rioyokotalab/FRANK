#include "hicma/operations/BLAS.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/classes/initialization_helpers/basis_tracker.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/pre_scheduler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"

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

define_method(
  void, trsm_omm,
  (const Hierarchical& A, Hierarchical& B, int uplo, int lr)
) {
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

define_method(void, trsm_omm, (const Dense& A, Dense& B, int uplo, int lr)) {
  timing::start("DTRSM");
  add_trsm_task(A, B, uplo, lr);
  timing::stop("DTRSM");
}

define_method(
  void, trsm_omm, (const Dense& A, NestedBasis& B, int uplo, int lr)
) {
  // TODO Only works for single layer!
  if (B.num_child_basis() != 0) abort();
  // Decouple basis
  // TODO Use different/more general tracker (like "copy")?
  // TODO Maybe this should not be in TRSM but in the main getrf?
  if (!matrix_is_tracked("decoupling", B.transfer_matrix)) {
    register_matrix("decoupling", B.transfer_matrix, Dense(B.transfer_matrix));
  }
  B.transfer_matrix = get_tracked_content("decoupling", B.transfer_matrix).share();
  trsm(A, B.transfer_matrix, uplo, lr);
}

define_method(void, trsm_omm, (const Matrix& A, LowRank& B, int uplo, int lr)) {
  switch (lr) {
  case TRSM_LEFT:
    trsm(A, B.U, uplo, lr);
    break;
  case TRSM_RIGHT:
    trsm(A, B.V, uplo, lr);
    break;
  }
}

define_method(
  void, trsm_omm,
  (const Hierarchical& A, Dense& B, int uplo, int lr)
) {
  Hierarchical BH(B, lr==TRSM_LEFT?A.dim[0]:1, lr==TRSM_LEFT?1:A.dim[1], false);
  trsm(A, BH, uplo, lr);
}

// Fallback default, abort with error message
define_method(
  void, trsm_omm,
  (
    const Matrix& A, Matrix& B,
    [[maybe_unused]] int uplo, [[maybe_unused]] int lr
  )
) {
  omm_error_handler("trsm", {A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
