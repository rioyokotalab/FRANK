#include "hicma/operations/BLAS.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/operations/arithmetic.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/counter.h"
#include "hicma/util/pre_scheduler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>


namespace hicma
{

void gemm(
  const Matrix& A, const Matrix& B, Matrix& C,
  double alpha, double beta,
  bool TransA, bool TransB
) {
  assert((TransA ? get_n_cols(A) : get_n_rows(A)) == get_n_rows(C));
  assert(
    (TransA ? get_n_rows(A) : get_n_cols(A))
    == (TransB ? get_n_cols(B) : get_n_rows(B))
  );
  assert((TransB ? get_n_rows(B) : get_n_cols(B)) == get_n_cols(C));
  gemm_omm(A, B, C, alpha, beta, TransA, TransB);
}

MatrixProxy gemm(
  const Matrix& A, const Matrix& B, double alpha, bool TransA, bool TransB
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
    double alpha, bool TransA, bool TransB
  )
) {
  // H H new
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
    double alpha, bool TransA, bool TransB
  )
) {
  Dense C(
    TransA ? get_n_cols(A) : get_n_rows(A),
    TransB ? get_n_rows(B) : get_n_cols(B)
  );
  gemm(A, B, C, alpha, 0, TransA, TransB);
  return std::move(C);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Dense& B, Dense& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // D D D
  timing::start("DGEMM");
  add_gemm_task(A, B, C, alpha, beta, TransA, TransB);
  timing::stop("DGEMM");
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Dense& B, Dense& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // LR D D
  Dense AS_basis_B = gemm(
    A.S, gemm(TransA ? A.U : A.V, B, alpha, TransA, TransB),
    1, TransA, false
  );
  gemm(TransA ? A.V : A.U, AS_basis_B, C, 1, beta, TransA, false);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const LowRank& B, Dense& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // D LR D
  Dense A_basis_BS = gemm(
    gemm(A, TransB ? B.V : B.U, alpha, TransA, TransB), B.S,
    1, false, TransB
  );
  gemm(A_basis_BS, TransB ? B.U : B.V, C, 1, beta, false, TransB);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, Dense& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // LR LR D
  // TODO Many optimizations possible here with shared basis
  // Even in non-shared case, UxS, SxV may be optimized across blocks!
  Dense Abasis_inner_matrices = gemm(
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
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // D D LR
  C.S *= beta;
  C += LowRank(gemm(A, B, alpha, TransA, TransB), C.rank);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Dense& B, LowRank& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // LR D LR
  // TODO Not implemented
  if (TransA) std::abort();
  Dense AVxB = gemm(A.V, B, alpha, false, TransB);
  C.S *= beta;
  LowRank AxB(A.U, A.S, AVxB);
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const LowRank& B, LowRank& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // D LR LR
  // TODO Not implemented
  if (TransB) std::abort();
  Dense AxBU = gemm(A, B.U, alpha, TransA, false);
  C.S *= beta;
  LowRank AxB(AxBU, B.S, B.V);
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, LowRank& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // LR LR LR
  // TODO Not implemented
  if (TransA || TransB) std::abort();
  assert(A.rank == B.rank);
  Dense SxVxU = gemm(A.S, gemm(A.V, B.U, alpha));
  Dense SxVxUxS = gemm(SxVxU, B.S);
  C.S *= beta;
  LowRank AxB(A.U, SxVxUxS, B.V);
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const LowRank& B, LowRank& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // H LR LR
  // TODO Not implemented
  if (TransA || TransB) std::abort();
  MatrixProxy AxBU = gemm(A, B.U, alpha, TransA, false);
  LowRank AxB(AxBU, B.S, B.V);
  C.S *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Hierarchical& B, LowRank& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // LR H LR
  // TODO Not implemented
  if (TransA || TransB) std::abort();
  MatrixProxy AVxB = gemm(A.V, B, alpha, false, TransB);
  LowRank AxB(A.U, A.S, AVxB);
  C.S *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, Hierarchical& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // LR LR H
  // TODO Not implemented
  if (TransA || TransB) std::abort();
  Dense SxVxUxS = gemm(gemm(A.S, gemm(A.V, B.U, alpha)), B.S);
  LowRank AxB(A.U, SxVxUxS, B.V);
  C *= beta;
  C += AxB;
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Hierarchical& B, LowRank& C,
    double alpha, double beta,
    bool TransA, bool TransB
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
  C = LowRank(CD, C.rank);
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Hierarchical& B, Hierarchical& C,
    double alpha, double beta,
    bool TransA, bool TransB
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
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // D D H
  // TODO Not implemented
  if (C.dim[0] == 1 && C.dim[1] == 1) std::abort();
  if (C.dim[1] == 1) {
    Hierarchical AH = split(A, TransA ? 1 : C.dim[0], TransA ? C.dim[0] : 1);
    gemm(AH, B, C, alpha, beta, TransA, TransB);
  } else if (C.dim[0] == 1) {
    Hierarchical BH = split(B, TransB ? C.dim[1] : 1, TransB ? 1 : C.dim[1]);
    gemm(A, BH, C, alpha, beta, TransA, TransB);
  } else {
    Hierarchical AH = split(A, TransA ? 1 : C.dim[0], TransA ? C.dim[0] : 1);
    Hierarchical BH = split(B, TransB ? C.dim[1] : 1, TransB ? 1 : C.dim[1]);
    gemm(AH, BH, C, alpha, beta, TransA, TransB);
  }
}

define_method(
  void, gemm_omm,
  (
    const Matrix& A, const Hierarchical& B, Hierarchical& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // D H H
  // LR H H
  // TODO Not implemented
  if (B.dim[0] == 1 && B.dim[1] == 1) std::abort();
  assert(B.dim[TransB ? 0 : 1] == C.dim[1]);
  if (B.dim[TransB ? 1 : 0] == 1 && C.dim[0] == 1) {
    for (int64_t j=0; j<B.dim[TransB ? 0 : 1]; ++j) {
      gemm(A, B[j], C[j], alpha, beta, TransA, TransB);
    }
  } else {
    Hierarchical AH = split(
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
    const Hierarchical& A, const Matrix& B, Hierarchical& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // H D H
  // H LR H
  // TODO Not implemented
  if (A.dim[0] == 1 && A.dim[1] == 1) std::abort();
  assert(A.dim[TransA ? 1 : 0] == C.dim[0]);
  if (A.dim[TransA ? 0 : 1] == 1 && C.dim[1] == 1) {
    for (int64_t i=0; i<A.dim[TransA ? 1 : 0]; ++i) {
      gemm(A[i], B, C[i], alpha, beta, TransA, TransB);
    }
  } else {
    Hierarchical BH = split(
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
    double alpha, double beta,
    bool TransA, bool TransB
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
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // D H D
  // LR H D
  // TODO Not implemented
  if (B.dim[0] == 1 && B.dim[1] == 1) std::abort();
  if (B.dim[TransB ? 1 : 0] == 1) {
    Hierarchical CH = split(C, 1, B.dim[TransB ? 0 : 1]);
    gemm(A, B, CH, alpha, beta, TransA, TransB);
  } else if (B.dim[TransB ? 0 : 1] == 1) {
    Hierarchical AH = split(
      A,
      TransA ? B.dim[TransB ? 1 : 0] : 1,
      TransA ? 1 : B.dim[TransB ? 1 : 0]
    );
    gemm(AH, B, C, alpha, beta, TransA, TransB);
  } else {
    Hierarchical AH = split(
      A,
      TransA ? B.dim[TransB ? 1 : 0] : 1,
      TransA ? 1 : B.dim[TransB ? 1 : 0]
    );
    Hierarchical CH = split(C, 1, B.dim[TransB ? 0 : 1]);
    gemm(AH, B, CH, alpha, beta, TransA, TransB);
  }
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Matrix& B, Dense& C,
    double alpha, double beta,
    bool TransA, bool TransB
  )
) {
  // H D D
  // H LR D
  // TODO Not implemented
  if (A.dim[0] == 1 && A.dim[1] == 1) std::abort();
  if (A.dim[TransA ? 0 : 1] == 1) {
    Hierarchical CH = split(C, A.dim[TransA ? 1 : 0], 1);
    gemm(A, B, CH, alpha, beta, TransA, TransB);
  } else if (A.dim[TransA ? 1 : 0] == 1) {
    Hierarchical BH = split(
      B,
      TransB ? 1 : A.dim[TransA ? 0 : 1],
      TransB ? A.dim[TransA ? 0 : 1] : 1
    );
    gemm(A, BH, C, alpha, beta, TransA, TransB);
  } else {
    Hierarchical BH = split(
      B,
      TransB ? 1 : A.dim[TransA ? 0 : 1],
      TransB ? A.dim[TransA ? 0 : 1] : 1
    );
    Hierarchical CH = split(C, A.dim[TransA ? 1 : 0], 1);
    gemm(A, BH, CH, alpha, beta, TransA, TransB);
  }
}

// Fallback default, abort with error message
define_method(
  void, gemm_omm,
  (const Matrix& A, const Matrix& B, Matrix& C, double, double, bool, bool)
) {
  omm_error_handler("gemm", {A, B, C}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
