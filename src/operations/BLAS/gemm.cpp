#include "FRANK/operations/BLAS.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/operations/arithmetic.h"
#include "FRANK/operations/BLAS.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"
#include "FRANK/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <cassert>
#include <cstdint>
#include <cstdlib>


namespace FRANK
{

declare_method(
  void, gemm_omm,
  (
    virtual_<const Matrix&>, virtual_<const Matrix&>, virtual_<Matrix&>,
    const double, const double, const bool, const bool
  )
)

void gemm(
  const Matrix& A, const Matrix& B, Matrix& C,
  const double alpha, const double beta,
  const bool TransA, const bool TransB
) {
  assert((TransA ? get_n_cols(A) : get_n_rows(A)) == get_n_rows(C));
  assert(
    (TransA ? get_n_rows(A) : get_n_cols(A))
    == (TransB ? get_n_cols(B) : get_n_rows(B))
  );
  assert((TransB ? get_n_rows(B) : get_n_cols(B)) == get_n_cols(C));
  gemm_omm(A, B, C, alpha, beta, TransA, TransB);
}

declare_method(
  MatrixProxy, gemm_omm,
  (
    virtual_<const Matrix&>, virtual_<const Matrix&>,
    const double, const bool, const bool
  )
)

MatrixProxy gemm(
  const Matrix& A, const Matrix& B,
  const double alpha, const bool TransA, const bool TransB
) {
  assert(
    (TransA ? get_n_rows(A) : get_n_cols(A))
    == (TransB ? get_n_cols(B) : get_n_rows(B))
  );
  return gemm_omm(A, B, alpha, TransA, TransB);
}

define_method(
  MatrixProxy, gemm_omm,
  (
    const Hierarchical& A, const Hierarchical& B,
    const double alpha, const bool TransA, const bool TransB
  )
) {
  // H H New(H)
  MatrixProxy C;
  if (A.dim[TransA ? 1 : 0] == 1 && B.dim[TransB ? 0 : 1] == 1) {
    // TODO Determine out type based on first pair? (A[0], A[1])
    C = Dense(
      TransA ? get_n_cols(A) : get_n_rows(A),
      TransB ? get_n_rows(B) : get_n_cols(B)
    );
    gemm(A, B, C, alpha, 0, TransA, TransB);
  } else {
    Hierarchical out(A.dim[TransA ? 1 : 0], B.dim[TransB ? 0 : 1]);
    // Note that first pair decides output type!
    for (int64_t i=0; i<out.dim[0]; ++i) {
      for (int64_t j=0; j<out.dim[1]; ++j) {
        out(i, j) = gemm(
          TransA ? A(0, i) : A(i, 0), TransB ? B(j, 0) : B(0, j),
          alpha, TransA, TransB
        );
        for (int64_t k=1; k<A.dim[TransA ? 0 : 1]; ++k) {
          gemm(
            TransA ? A(k, i) : A(i, k), TransB ? B(j, k) : B(k, j), out(i, j),
            alpha, 1, TransA, TransB
          );
        }
      }
    }
    C = std::move(out);
  }
  return C;
}

define_method(
  MatrixProxy, gemm_omm,
  (
    const Matrix& A, const Matrix& B,
    const double alpha, const bool TransA, const bool TransB
  )
) {
  // * * New(D)
  Dense C(
    TransA ? get_n_cols(A) : get_n_rows(A),
    TransB ? get_n_rows(B) : get_n_cols(B)
  );
  gemm(A, B, C, alpha, 0, TransA, TransB);
  return C;
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Dense& B, Dense& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // D D D
  const int64_t k = TransA ? A.dim[0] : A.dim[1];
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

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Dense& B, Dense& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // LR D D
  const Dense AS_basis_B = gemm(
    A.S, gemm(TransA ? A.U : A.V, B, alpha, TransA, TransB),
    1, TransA, false
  );
  gemm(TransA ? A.V : A.U, AS_basis_B, C, 1, beta, TransA, false);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const LowRank& B, Dense& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // D LR D
  const Dense A_basis_BS = gemm(
    gemm(A, TransB ? B.V : B.U, alpha, TransA, TransB), B.S,
    1, false, TransB
  );
  gemm(A_basis_BS, TransB ? B.U : B.V, C, 1, beta, false, TransB);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, Dense& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // LR LR D
  // TODO Many optimizations possible
  // Even in non-shared case, UxS, SxV may be optimized across blocks!
  const Dense Abasis_inner_matrices = gemm(
    TransA ? A.V : A.U, gemm(
      gemm(
        A.S, gemm(TransA ? A.U : A.V, TransB ? B.V : B.U, alpha, TransA, TransB),
        1, TransA, false
      ), B.S, 1, false, TransB
    ), 1, TransA, false
  );
  gemm(Abasis_inner_matrices, TransB ? B.U : B.V, C, 1, beta, false, TransB);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Dense& B, LowRank& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // D D LR
  C.S *= beta;
  const bool use_eps = (C.eps != 0.0);
  if(use_eps)
    C += LowRank(gemm(A, B, alpha, TransA, TransB), C.eps);
  else
    C += LowRank(gemm(A, B, alpha, TransA, TransB), C.rank);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Dense& B, LowRank& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // LR D LR
  const Dense AxB_U = TransA ? transpose(A.V) : shallow_copy(A.U);
  const Dense AxB_S = TransA ? transpose(A.S) : shallow_copy(A.S);
  const Dense AxB_V = gemm(TransA ? A.U : A.V, B, alpha, TransA, TransB);
  const LowRank AxB(AxB_U, AxB_S, AxB_V, false);
  C *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const LowRank& B, LowRank& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // D LR LR
  const Dense AxB_U = gemm(A, TransB ? B.V : B.U, alpha, TransA, TransB);
  const Dense AxB_S = TransB ? transpose(B.S) : shallow_copy(B.S);
  const Dense AxB_V = TransB ? transpose(B.U) : shallow_copy(B.V);
  const LowRank AxB(AxB_U, AxB_S, AxB_V, false);
  C *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, LowRank& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // LR LR LR
  const Dense AxB_U = TransA ? transpose(A.V) : shallow_copy(A.U);
  const Dense AxB_S = gemm(
      gemm(A.S, TransA ? A.U : A.V, 1, TransA, TransA),
      gemm(TransB ? B.V : B.U, B.S, 1, TransB, TransB),
      alpha
  );
  const Dense AxB_V = TransB ? transpose(B.U) : shallow_copy(B.V);
  const LowRank AxB(AxB_U, AxB_S, AxB_V, false);
  C *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const LowRank& B, LowRank& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // H LR LR
  const Dense AxB_U = gemm(A, TransB ? B.V : B.U, alpha, TransA, TransB);
  const Dense AxB_S = TransB ? transpose(B.S) : shallow_copy(B.S);
  const Dense AxB_V = TransB ? transpose(B.U) : shallow_copy(B.V);
  const LowRank AxB(AxB_U, AxB_S, AxB_V, false);
  C *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Hierarchical& B, LowRank& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // D H LR
  const Hierarchical HA = split(A,
                                TransA ? B.dim[TransB ? 1 : 0] : 1,
                                TransA ? 1 : B.dim[TransB ? 1 : 0],
                                false);
  gemm(HA, B, C, alpha, beta, TransA, TransB);
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Dense& B, LowRank& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // H D LR
  const Hierarchical HB = split(B,
                          TransB ? 1 : A.dim[TransA ? 0 : 1],
                          TransB ? A.dim[TransA ? 0 : 1] : 1,
                          false);
  gemm(A, HB, C, alpha, beta, TransA, TransB);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Hierarchical& B, LowRank& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // LR H LR
  const Dense AxB_U = TransA ? transpose(A.V) : shallow_copy(A.U);
  const Dense AxB_S = TransA ? transpose(A.S) : shallow_copy(A.S);
  const Dense AxB_V = gemm(TransA ? A.U : A.V, B, alpha, TransA, TransB);
  const LowRank AxB(AxB_U, AxB_S, AxB_V, false);
  C *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const LowRank& B, Hierarchical& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // D LR H
  const Dense AxB_U = gemm(A, TransB ? B.V : B.U, alpha, TransA, TransB);
  const Dense AxB_S = TransB ? transpose(B.S) : shallow_copy(B.S);
  const Dense AxB_V = TransB ? transpose(B.U) : shallow_copy(B.V);
  const LowRank AxB(AxB_U, AxB_S, AxB_V, false);
  C *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Dense& B, Hierarchical& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // LR D H
  const Dense AxB_U = TransA ? transpose(A.V) : shallow_copy(A.U);
  const Dense AxB_S = TransA ? transpose(A.S) : shallow_copy(A.S);
  const Dense AxB_V = gemm(TransA ? A.U : A.V, B, alpha, TransA, TransB);
  const LowRank AxB(AxB_U, AxB_S, AxB_V, false);
  C *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, Hierarchical& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // LR LR H
  const Dense AxB_U = TransA ? transpose(A.V) : shallow_copy(A.U);
  const Dense AxB_S = gemm(
      gemm(A.S, TransA ? A.U : A.V, 1, TransA, TransA),
      gemm(TransB ? B.V : B.U, B.S, 1, TransB, TransB),
      alpha
  );
  const Dense AxB_V = TransB ? transpose(B.U) : shallow_copy(B.V);
  const LowRank AxB(AxB_U, AxB_S, AxB_V, false);
  C *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Hierarchical& B, LowRank& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // H H LR
  /*
    Making a Hierarchical out of C might be better
    But LowRank(Hierarchical, rank) constructor is needed
      Hierarchical CH(C);
      gemm(A, B, CH, alpha, beta);
      C = LowRank(CH, rank);
  */
  Dense CD(C);
  gemm(A, B, CD, alpha, beta, TransA, TransB);
  const bool use_eps = (C.eps != 0.0);
  if(use_eps)
    C = LowRank(CD, C.eps);
  else
    C = LowRank(CD, C.rank);
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Hierarchical& B, Hierarchical& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // H H H
  assert(A.dim[TransA ? 1 : 0] == C.dim[0]);
  assert(A.dim[TransA ? 0 : 1] == B.dim[TransB ? 1 : 0]);
  assert(B.dim[TransB ? 0 : 1] == C.dim[1]);
  for (int64_t i=0; i<C.dim[0]; i++) {
    for (int64_t j=0; j<C.dim[1]; j++) {
      for (int64_t k=0; k<A.dim[TransA ? 0 : 1]; k++) {
        gemm(
          TransA ? A(k, i) : A(i, k), TransB ? B(j, k) : B(k, j), C(i, j),
          alpha, k==0 ? beta : 1, TransA, TransB
        );
      }
    }
  }
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Dense& B, Hierarchical& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // D D H
  // TODO Not implemented
  if (C.dim[0] == 1 && C.dim[1] == 1) std::abort();
  if (C.dim[1] == 1) {
    const Hierarchical AH = split(A, TransA ? 1 : C.dim[0], TransA ? C.dim[0] : 1);
    gemm(AH, B, C, alpha, beta, TransA, TransB);
  } else if (C.dim[0] == 1) {
    const Hierarchical BH = split(B, TransB ? C.dim[1] : 1, TransB ? 1 : C.dim[1]);
    gemm(A, BH, C, alpha, beta, TransA, TransB);
  } else {
    const Hierarchical AH = split(A, TransA ? 1 : C.dim[0], TransA ? C.dim[0] : 1);
    const Hierarchical BH = split(B, TransB ? C.dim[1] : 1, TransB ? 1 : C.dim[1]);
    gemm(AH, BH, C, alpha, beta, TransA, TransB);
  }
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Hierarchical& B, Hierarchical& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // LR H H
  assert(B.dim[TransB ? 0 : 1] == C.dim[1]);
  if (B.dim[0] == 1 && B.dim[1] == 1) {
    gemm(A, B(0, 0), C, alpha, beta, TransA, TransB);
  }
  else if (B.dim[TransB ? 1 : 0] == 1 && C.dim[0] == 1) {
    for (int64_t j=0; j<B.dim[TransB ? 0 : 1]; ++j) {
      gemm(A, B[j], C[j], alpha, beta, TransA, TransB);
    }
  }
  else {
    Dense AVxB = gemm(A.V, B, alpha, false, TransB);
    C *= beta;
    LowRank AxB(A.U, A.S, AVxB, false);
    C += AxB;
  }
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Hierarchical& B, Hierarchical& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // D H H
  assert(B.dim[TransB ? 0 : 1] == C.dim[1]);
  if (B.dim[0] == 1 && B.dim[1] == 1) {
    gemm(A, B(0, 0), C, alpha, beta, TransA, TransB);
  }
  else if (B.dim[TransB ? 1 : 0] == 1 && C.dim[0] == 1) {
    for (int64_t j=0; j<B.dim[TransB ? 0 : 1]; ++j) {
      gemm(A, B[j], C[j], alpha, beta, TransA, TransB);
    }
  }
  else {
    const Hierarchical AH = split(
      A,
      TransA ? B.dim[TransB ? 1 : 0] : C.dim[0],
      TransA ? C.dim[0] : B.dim[TransB ? 1 : 0]
    );
    gemm(AH, B, C, alpha, beta, TransA, TransB);
  }
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const LowRank& B, Hierarchical& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // H LR H
  if (A.dim[0] == 1 && A.dim[1] == 1) {
    gemm(A(0, 0), B, C, alpha, beta, TransA, TransB);
  }
  else if (A.dim[TransA ? 0 : 1] == 1 && C.dim[1] == 1) {
    for (int64_t i=0; i<A.dim[TransA ? 1 : 0]; ++i) {
      gemm(A[i], B, C[i], alpha, beta, TransA, TransB);
    }
  }
  else {
    Dense AxBU = gemm(A, B.U, alpha, false, TransB);
    C *= beta;
    LowRank AxB(AxBU, B.S, B.V, false);
    C += AxB;
  }
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Dense& B, Hierarchical& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // H D H
  if (A.dim[0] == 1 && A.dim[1] == 1) {
    gemm(A(0, 0), B, C, alpha, beta, TransA, TransB);
  }
  else if (A.dim[TransA ? 0 : 1] == 1 && C.dim[1] == 1) {
    for (int64_t i=0; i<A.dim[TransA ? 1 : 0]; ++i) {
      gemm(A[i], B, C[i], alpha, beta, TransA, TransB);
    }
  }
  else {
    const Hierarchical BH = split(
      B,
      TransB ? C.dim[1] : A.dim[TransA ? 0 : 1],
      TransB ? A.dim[TransA ? 0 : 1] : C.dim[1]
    );
    gemm(A, BH, C, alpha, beta, TransA, TransB);
  }
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Hierarchical& B, Dense& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // H H D
  assert(A.dim[TransA ? 0 : 1] == B.dim[TransB ? 1 : 0]);
  if (A.dim[TransA ? 1 : 0] == 1 && B.dim[TransB ? 0 : 1] == 1) {
    for (int64_t k = 0; k < A.dim[TransA ? 0 : 1]; ++k) {
      gemm(A[k], B[k], C, alpha, k==0 ? beta : 1, TransA, TransB);
    }
  } else {
    Hierarchical CH = split(C, A.dim[TransA ? 1 : 0], B.dim[TransB ? 0 : 1]);
    gemm(A, B, CH, alpha, beta, TransA, TransB);
  }
}

define_method(
  void, gemm_omm,
  (
    const Matrix& A, const Hierarchical& B, Dense& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // D H D
  // LR H D
  if (B.dim[0] == 1 && B.dim[1] == 1) {
    gemm(A, B(0, 0), C, alpha, beta, TransA, TransB);
  }
  /*else if (B.dim[TransB ? 1 : 0] == 1) {
    Hierarchical CH = split(C, 1, B.dim[TransB ? 0 : 1]);
    gemm(A, B, CH, alpha, beta, TransA, TransB);
  }
  else if (B.dim[TransB ? 0 : 1] == 1) {
    const Hierarchical AH = split(
      A,
      TransA ? B.dim[TransB ? 1 : 0] : 1,
      TransA ? 1 : B.dim[TransB ? 1 : 0]
    );
    gemm(AH, B, C, alpha, beta, TransA, TransB);
  }*/
  else {
    /*const Hierarchical AH = split(
      A,
      TransA ? B.dim[TransB ? 1 : 0] : 1,
      TransA ? 1 : B.dim[TransB ? 1 : 0]
    );*/
    Hierarchical CH = split(C, 1, B.dim[TransB ? 0 : 1]);
    gemm(A, B, CH, alpha, beta, TransA, TransB);
  }
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Matrix& B, Dense& C,
    const double alpha, const double beta,
    const bool TransA, const bool TransB
  )
) {
  // H D D
  // H LR D
  if (A.dim[0] == 1 && A.dim[1] == 1) {
    gemm(A(0, 0), B, C, alpha, beta, TransA, TransB);
  }
  /*else if (A.dim[TransA ? 0 : 1] == 1) {
    Hierarchical CH = split(C, A.dim[TransA ? 1 : 0], 1);
    gemm(A, B, CH, alpha, beta, TransA, TransB);
  }
  else if (A.dim[TransA ? 1 : 0] == 1) {
    const Hierarchical BH = split(
      B,
      TransB ? 1 : A.dim[TransA ? 0 : 1],
      TransB ? A.dim[TransA ? 0 : 1] : 1
    );
    gemm(A, BH, C, alpha, beta, TransA, TransB);
  }*/
  else {
    /*const Hierarchical BH = split(
      B,
      TransB ? 1 : A.dim[TransA ? 0 : 1],
      TransB ? A.dim[TransA ? 0 : 1] : 1
    );*/
    Hierarchical CH = split(C, A.dim[TransA ? 1 : 0], 1);
    gemm(A, B, CH, alpha, beta, TransA, TransB);
  }
}

// Fallback default, abort with error message
define_method(
  void, gemm_omm,
  (const Matrix& A, const Matrix& B, Matrix& C,
   const double, const double, const bool, const bool)
) {
  omm_error_handler("gemm", {A, B, C}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
