#include "hicma/operations/BLAS.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/shared_basis.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/operations/arithmetic.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/counter.h"
#include "hicma/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif
#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>


namespace hicma
{

declare_method(
  void, gemm_trans_omm,
  (
    virtual_<const Matrix&>, virtual_<const Matrix&>, virtual_<Matrix&>,
    bool, bool, double, double
  )
)

void gemm(
  const Matrix& A, const Matrix& B, Matrix& C,
  bool TransA, bool TransB,
  double alpha, double beta
) {
  assert(
    (TransA ? get_n_cols(A) : get_n_rows(A))
    == TransB ? get_n_cols(C) : get_n_rows(C)
  );
  assert(
    (TransA ? get_n_rows(A) : get_n_cols(A))
    == TransB ? get_n_cols(B) : get_n_rows(B)
  );
  assert(
    (TransA ? get_n_rows(B) : get_n_cols(B))
    == TransB ? get_n_rows(C) : get_n_cols(C)
  );
  gemm_trans_omm(A, B, C, TransA, TransB, alpha, beta);
}

define_method(
  void, gemm_trans_omm,
  (
    const Dense& A, const Dense& B, Dense& C,
    bool TransA, bool TransB,
    double alpha, double beta
  )
) {
  timing::start("DGEMM");
  if (B.dim[1] == 1) {
    cblas_dgemv(
      CblasRowMajor,
      CblasNoTrans,
      A.dim[0], A.dim[1],
      alpha,
      &A, A.stride,
      &B, B.stride,
      beta,
      &C, B.stride
    );
  }
  else {
    int64_t k = TransA ? A.dim[0] : A.dim[1];
    cblas_dgemm(
      CblasRowMajor,
      TransA?CblasTrans:CblasNoTrans, TransB?CblasTrans:CblasNoTrans,
      C.dim[0], C.dim[1], k,
      alpha,
      &A, A.stride,
      &B, B.stride,
      beta,
      &C, C.stride
    );
  }
  timing::stop("DGEMM");
}

// Fallback default, abort with error message
define_method(
  void, gemm_trans_omm,
  (
    const Matrix& A, const Matrix& B, Matrix& C,
    [[maybe_unused]] bool TransA, [[maybe_unused]] bool TransB,
    [[maybe_unused]] double alpha, [[maybe_unused]] double beta
  )
) {
  omm_error_handler("gemm_trans", {A, B, C}, __FILE__, __LINE__);
  std::abort();
}

void gemm(
  const Matrix& A, const Matrix& B, Matrix& C,
  double alpha, double beta
) {
  assert(get_n_rows(A) == get_n_rows(C));
  assert(get_n_cols(A) == get_n_rows(B));
  assert(get_n_cols(B) == get_n_cols(C));
  // TODO Define special cases for beta=0, beta=1, alpha=1?
  gemm_omm(A, B, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  if (alpha == 1 && beta == 1) {
    gemm_push(A, B, C);
  }
  else {
    gemm(A, B, C, false, false, alpha, beta);
  }
}

define_method(
  void, gemm_omm,
  (
    const SharedBasis& A, const SharedBasis& B, Matrix& C,
    double alpha, double beta
  )
) {
  gemm(*A.get_ptr(), *B.get_ptr(), C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const SharedBasis& A, const Matrix& B, Matrix& C,
    double alpha, double beta
  )
) {
  gemm(*A.get_ptr(), B, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Matrix& A, const SharedBasis& B, Matrix& C,
    double alpha, double beta
  )
) {
  gemm(A, *B.get_ptr(), C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  Dense SxVxB = gemm(A.S, gemm(A.V, B));
  gemm(A.U, SxVxB, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const LowRank& B, Dense& C,
    double alpha, double beta
  )
) {
  Dense AxUxS = gemm(gemm(A, B.U), B.S);
  gemm(AxUxS, B.V, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, Dense& C,
    double alpha, double beta
  )
) {
  Dense UxSxVxUxS = gemm(A.U, gemm(gemm(A.S, gemm(A.V, B.U)), B.S));
  gemm(UxSxVxUxS, B.V, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Dense& B, LowRank& C,
    double alpha, double beta
  )
) {
  C.S *= beta;
  C += LowRank(gemm(A, B, alpha), C.rank);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Dense& B, LowRank& C,
    double alpha, double beta
  )
) {
  Dense AVxB = gemm(A.V, B, alpha);
  LowRank AxB(A.U, A.S, AVxB);
  C.S *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const LowRank& B, LowRank& C,
    double alpha, double beta
  )
) {
  Dense AxBU = gemm(A, B.U, alpha);
  LowRank AxB(AxBU, B.S, B.V);
  C.S *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Hierarchical& B, LowRank& C,
    double alpha, double beta
  )
) {
  Dense AVxB = gemm(A.V, B, alpha);
  LowRank AxB(A.U, A.S, AVxB);
  C.S *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const LowRank& B, LowRank& C,
    double alpha, double beta
  )
) {
  Dense AxBU = gemm(A, B.U, alpha);
  LowRank AxB(AxBU, B.S, B.V);
  C.S *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Hierarchical& B, LowRank& C,
    double alpha, double beta
  )
) {
  /*
    Making a Hierarchical out of C might be better
    But LowRank(Hierarchical, rank) constructor is needed
    Hierarchical CH(C);
      gemm(A, B, CH, alpha, beta);
    C = LowRank(CH, rank);
  */
  Dense CD(C);
  gemm(A, B, CD, alpha, beta);
  C = LowRank(CD, C.rank);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, LowRank& C,
    double alpha, double beta
  )
) {
  assert(A.rank == B.rank);
  Dense SxVxUxS = gemm(gemm(A.S, gemm(A.V, B.U)), B.S, alpha);
  LowRank AxB(A.U, SxVxUxS, B.V);
  C.S *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Hierarchical& B, Hierarchical& C,
    double alpha, double beta
  )
) {
  assert(C.dim[0] == A.dim[0]);
  assert(C.dim[1] == B.dim[1]);
  assert(A.dim[1] == B.dim[0]);
  for (int64_t i=0; i<C.dim[0]; i++) {
    for (int64_t j=0; j<C.dim[1]; j++) {
      gemm(A(i,0), B(0,j), C(i,j), alpha, beta);
      for (int64_t k=1; k<A.dim[1]; k++) {
        gemm(A(i,k), B(k,j), C(i,j), alpha, 1);
      }
    }
  }
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Dense& B, Hierarchical& C,
    double alpha, double beta
  )
) {
  Hierarchical AH(A, C.dim[0], 1, false);
  Hierarchical BH(B, 1, C.dim[1], false);
  gemm(AH, BH, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, Hierarchical& C,
    double alpha, double beta
  )
) {
  Dense SxVxUxS = gemm(gemm(A.S, gemm(A.V, B.U)), B.S, alpha);
  LowRank AxB(A.U, SxVxUxS, B.V);
  C *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Dense& B, Hierarchical& C,
    double alpha, double beta
  )
) {
  assert(A.dim[0] == C.dim[0]);
  Hierarchical BH(B, A.dim[1], C.dim[1], false);
  gemm(A, BH, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const LowRank& B, Hierarchical& C,
    double alpha, double beta
  )
) {
  assert(A.dim[0] == C.dim[0]);
  Hierarchical BH(B, A.dim[1], C.dim[1], false);
  gemm(A, BH, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Hierarchical& B, Hierarchical& C,
    double alpha, double beta
  )
) {
  assert(B.dim[1] == C.dim[1]);
  Hierarchical AH(A, C.dim[0], B.dim[0], false);
  gemm(AH, B, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Hierarchical& B, Hierarchical& C,
    double alpha, double beta
  )
) {
  assert(B.dim[1] == C.dim[1]);
  Hierarchical AH(A, C.dim[0], B.dim[0], false);
  gemm(AH, B, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Hierarchical& B, Dense& C,
    double alpha, double beta
  )
) {
  assert(A.dim[1] == B.dim[0]);
  Hierarchical CH(C, A.dim[0], B.dim[1], false);
  gemm(A, B, CH, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Hierarchical& B, Dense& C,
    double alpha, double beta
  )
) {
  Hierarchical AH(A, 1, B.dim[0], false);
  Hierarchical CH(C, 1, B.dim[1], false);
  gemm(AH, B, CH, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const LowRank& B, Dense& C,
    double alpha, double beta
  )
) {
  Hierarchical BH(B, A.dim[1], 1, false);
  Hierarchical CH(C, A.dim[0], 1, false);
  gemm(A, BH, CH, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Hierarchical& B, Dense& C,
    double alpha, double beta
  )
) {
  Hierarchical AH(A, 1, B.dim[0], false);
  Hierarchical CH(C, 1, B.dim[1], false);
  gemm(AH, B, CH, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  Hierarchical BH(B, A.dim[1], 1, false);
  Hierarchical CH(C, A.dim[0], 1, false);
  gemm(A, BH, CH, alpha, beta);
}

// Fallback default, abort with error message
define_method(
  void, gemm_omm,
  (
    const Matrix& A, const Matrix& B, Matrix& C,
    [[maybe_unused]] double alpha, [[maybe_unused]] double beta
  )
) {
  omm_error_handler("gemm", {A, B, C}, __FILE__, __LINE__);
  std::abort();
}

Dense gemm(
  const Matrix& A, const Matrix& B, double alpha, bool TransA, bool TransB
) {
  assert(
    (TransA ? get_n_rows(A) : get_n_cols(A))
    == TransB ? get_n_cols(B) : get_n_rows(B)
  );
  return gemm_omm(A, B, alpha, TransA, TransB);
}

define_method(
  Dense, gemm_omm,
  (
    const Dense& A, const Dense& B,
    double alpha, bool TransA, bool TransB
  )
) {
  Dense out(A.dim[TransA ? 1 : 0], B.dim[TransB ? 0 : 1]);
  gemm(A, B, out, TransA, TransB, alpha, 0);
  return out;
}

define_method(
  Dense, gemm_omm,
  (
    const SharedBasis& A, const SharedBasis& B,
    double alpha, bool TransA, bool TransB
  )
) {
  return gemm(*A.get_ptr(), *B.get_ptr(), alpha, TransA, TransB);
}

define_method(
  Dense, gemm_omm,
  (
    const SharedBasis& A, const Matrix& B,
    double alpha, bool TransA, bool TransB
  )
) {
  return gemm(*A.get_ptr(), B, alpha, TransA, TransB);
}

define_method(
  Dense, gemm_omm,
  (
    const Matrix& A, const SharedBasis& B,
    double alpha, bool TransA, bool TransB
  )
) {
  return gemm(A, *B.get_ptr(), alpha, TransA, TransB);
}

define_method(
  Dense, gemm_omm,
  (
    const Hierarchical& A, const Dense& B,
    double alpha, [[maybe_unused]] bool TransA, [[maybe_unused]] bool TransB
  )
) {
  // TODO Implement with transposed allowed
  assert(TransA == false);
  assert(TransB == false);
  Dense out(get_n_rows(A), B.dim[1]);
  Hierarchical outH(out, A.dim[0], 1, false);
  Hierarchical BH(B, A.dim[1], 1, false);
  gemm(A, BH, outH, alpha, 0);
  return out;
}

define_method(
  Dense, gemm_omm,
  (
    const Dense& A, const Hierarchical& B,
    double alpha, [[maybe_unused]] bool TransA, [[maybe_unused]] bool TransB
  )
) {
  // TODO Implement with transposed allowed
  assert(TransA == false);
  assert(TransB == false);
  Dense out(A.dim[0], get_n_cols(B));
  Hierarchical outH(out, 1, B.dim[1], false);
  Hierarchical AH(A, 1, B.dim[0], false);
  gemm(AH, B, outH, alpha, 0);
  return out;
}

define_method(
  Dense, gemm_omm,
  (
    const Matrix& A, const Matrix& B,
    [[maybe_unused]] double alpha,
    [[maybe_unused]] bool TransA, [[maybe_unused]] bool TransB
  )
) {
  omm_error_handler("gemm", {A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
