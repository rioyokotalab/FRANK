#include "hicma/operations/BLAS/gemm.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/low_rank_view.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/no_copy_split.h"
#include "hicma/classes/uniform_hierarchical.h"
#include "hicma/operations/misc/addition.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/util/timer.h"
#include "hicma/util/counter.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/operations/misc/get_dim.h"

#include <cassert>
#include <iostream>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif
#include "yorel/yomm2/cute.hpp"

namespace hicma
{

declare_method(
  void, gemm_trans_omm,
  (
    virtual_<const Node&>, virtual_<const Node&>, virtual_<Node&>,
    bool, bool, double, double
  )
);

void gemm(
  const Node& A, const Node& B, Node& C,
  bool TransA, bool TransB,
  double alpha, double beta
) {
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
    int k = TransA ? A.dim[0] : A.dim[1];
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
    const Node& A, const Node& B, Node& C,
    [[maybe_unused]] bool TransA, [[maybe_unused]] bool TransB,
    [[maybe_unused]] double alpha, [[maybe_unused]] double beta
  )
) {
  std::cerr << "gemm_trans(";
  std::cerr << A.type() << "," << B.type() << "," << C.type();
  std::cerr << ") undefined." << std::endl;
  abort();
}

void gemm(
  const Node& A, const Node& B, Node& C,
  double alpha, double beta
) {
  gemm_omm(A, B, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  assert(C.dim[0] == A.dim[0]);
  assert(A.dim[1] == B.dim[0]);
  assert(C.dim[1] == B.dim[1]);
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
    const LowRank& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  Dense VxB(A.rank, B.dim[1]);
  gemm(A.V(), B, VxB, 1, 0);
  Dense SxVxB(A.rank, B.dim[1]);
  gemm(A.S(), VxB, SxVxB, 1, 0);
  gemm(A.U(), SxVxB, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const LowRank& B, Dense& C,
    double alpha, double beta
  )
) {
  Dense AxU(C.dim[0], B.rank);
  gemm(A, B.U(), AxU, 1, 0);
  Dense AxUxS(C.dim[0], B.rank);
  gemm(AxU, B.S(), AxUxS, 1, 0);
  gemm(AxUxS, B.V(), C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, Dense& C,
    double alpha, double beta
  )
) {
  Dense VxU(A.rank, B.rank);
  gemm(A.V(), B.U(), VxU, 1, 0);
  Dense SxVxU(A.rank, B.rank);
  gemm(A.S(), VxU, SxVxU, 1, 0);
  Dense SxVxUxS(A.rank, B.rank);
  gemm(SxVxU, B.S(), SxVxUxS, 1, 0);
  Dense UxSxVxUxS(A.dim[0], B.rank);
  gemm(A.U(), SxVxUxS, UxSxVxUxS, 1, 0);
  gemm(UxSxVxUxS, B.V(), C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Dense& B, LowRank& C,
    double alpha, double beta
  )
) {
  assert(C.dim[0] == A.dim[0]);
  assert(A.dim[1] == B.dim[0]);
  assert(C.dim[1] == B.dim[1]);
  Dense AB(C.dim[0], C.dim[1]);
  gemm(A, B, AB, alpha, 0);
  C.S() *= beta;
  C += LowRank(AB, C.rank);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Dense& B, LowRank& C,
    double alpha, double beta
  )
) {
  // TODO could be optimized to copy less with LowRankView!
  LowRank AVxB(A);
  gemm(A.V(), B, AVxB.V(), alpha, 0);
  C.S() *= beta;
  C += AVxB;
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const LowRank& B, LowRank& C,
    double alpha, double beta
  )
) {
  // TODO could be optimized to copy less with LowRankView!
  LowRank AxBU(B);
  gemm(A, B.U(), AxBU.U(), alpha, 0);
  C.S() *= beta;
  C += AxBU;
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, LowRank& C,
    double alpha, double beta
  )
) {
  assert(A.rank == B.rank);
  LowRankView AxB(A, A);
  AxB.V() = B.V();
  AxB.col_range = B.col_range;
  AxB.dim[1] = B.dim[1];
  Dense S(A.rank, B.rank);
  gemm(A.V(), B.U(), S, 1, 0);
  Dense SxVxU(A.rank, B.rank);
  gemm(A.S(), S, SxVxU, 1, 0);
  gemm(SxVxU, B.S(), S, alpha, 0);
  AxB.S() = S;
  C.S() *= beta;
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
  for (int i=0; i<C.dim[0]; i++) {
    for (int j=0; j<C.dim[1]; j++) {
      gemm(A(i,0), B(0,j), C(i,j), alpha, beta);
      for (int k=1; k<A.dim[1]; k++) {
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
  NoCopySplit AH(A, C.dim[0], 1);
  NoCopySplit BH(B, 1, C.dim[1]);
  gemm(AH, BH, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const LowRank& B, Hierarchical& C,
    double alpha, double beta
  )
) {
  LowRankView AxB(A, A);
  AxB.V() = B.V();
  AxB.col_range = B.col_range;
  AxB.dim[1] = B.dim[1];
  Dense S(A.rank, B.rank);
  gemm(A.V(), B.U(), S, 1, 0);
  Dense SxVxU(A.rank, B.rank);
  gemm(A.S(), S, SxVxU, 1, 0);
  gemm(SxVxU, B.S(), S, alpha, 0);
  AxB.S() = S;
  AxB.S() *= beta;
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
  NoCopySplit BH(B, A.dim[1], C.dim[1]);
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
  NoCopySplit BH(B, A.dim[1], C.dim[1]);
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
  NoCopySplit AH(A, C.dim[0], B.dim[0]);
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
  NoCopySplit AH(A, C.dim[0], B.dim[0]);
  gemm(AH, B, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const LowRank& B, LowRank& C,
    double alpha, double beta
  )
) {
  // TODO could be optimized to copy less with LowRankView!
  LowRank B_copy(B);
  gemm(A, B.U(), B_copy.U(), 1, 0);
  B_copy.S() *= alpha;
  C.S() *= beta;
  C += B_copy;
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Hierarchical& B, LowRank& C,
    double alpha, double beta
  )
) {
  // TODO could be optimized to copy less with LowRankView!
  LowRank A_copy(A);
  gemm(A.V(), B, A_copy.V(), 1, 0);
  A_copy.S() *= alpha;
  C.S() *= beta;
  C += A_copy;
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
    const Hierarchical& A, const Hierarchical& B, Dense& C,
    double alpha, double beta
  )
) {
  assert(A.dim[1] == B.dim[0]);
  NoCopySplit CH(C, A.dim[0], B.dim[1]);
  gemm(A, B, CH, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Dense& A, const Hierarchical& B, Dense& C,
    double alpha, double beta
  )
) {
  NoCopySplit AH(A, 1, B.dim[0]);
  NoCopySplit CH(C, 1, B.dim[1]);
  gemm(AH, B, CH, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRank& A, const Hierarchical& B, Dense& C,
    double alpha, double beta
  )
) {
  NoCopySplit AH(A, 1, B.dim[0]);
  NoCopySplit CH(C, 1, B.dim[1]);
  gemm(AH, B, CH, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const Hierarchical& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  NoCopySplit BH(B, A.dim[1], 1);
  NoCopySplit CH(C, A.dim[0], 1);
  gemm(A, BH, CH, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRankShared& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  Dense VxB(A.rank, B.dim[1]);
  gemm(A.V, B, VxB, 1, 0);
  Dense SxVxB(A.rank, B.dim[1]);
  gemm(A.S, VxB, SxVxB, 1, 0);
  gemm(A.U, SxVxB, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRankShared& A, const LowRankShared& B, Dense& C,
    double alpha, double beta
  )
) {
  // TODO Exactly the same as gemm(LR, LR, D)! Consider making LRS a child of LR
  Dense VxU(A.rank, B.rank);
  gemm(A.V, B.U, VxU, 1, 0);
  Dense SxVxU(A.rank, B.rank);
  gemm(A.S, VxU, SxVxU, 1, 0);
  Dense SxVxUxS(A.rank, B.rank);
  gemm(SxVxU, B.S, SxVxUxS, 1, 0);
  Dense UxSxVxUxS(A.dim[0], B.rank);
  gemm(A.U, SxVxUxS, UxSxVxUxS, 1, 0);
  gemm(UxSxVxUxS, B.V, C, alpha, beta);
}

define_method(
  void, gemm_omm,
  (
    const LowRankShared& A, const LowRankShared& B, LowRankShared& C,
    double alpha, double beta
  )
) {
  assert(C.U == A.U);
  assert(C.V == B.V);
  Dense VxU(A.rank, B.rank);
  gemm(A.V, B.U, VxU, 1, 0);
  Dense SxVxU(A.rank, B.rank);
  gemm(A.S, VxU, SxVxU, 1, 0);
  gemm(SxVxU, B.S, C.S, alpha, beta);
}

declare_method(
  void, gemm_regular_only_omm,
  (
    virtual_<const Node&>, virtual_<const Node&>, virtual_<Node&>,
    double, double
  )
);

void gemm_regular_only(
  const Node& A, const Node& B, Node& C, double alpha, double beta
) {
  gemm_regular_only_omm(A, B, C, alpha, beta);
}

define_method(
  void, gemm_regular_only_omm,
  (
    const UniformHierarchical& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  gemm(A, B, C, alpha, beta);
}

define_method(
  void, gemm_regular_only_omm,
  (
    const UniformHierarchical& A, const Hierarchical& B, Hierarchical& C,
    double alpha, double beta
  )
) {
  for (int i=0; i<C.dim[0]; i++) {
    for (int j=0; j<C.dim[1]; j++) {
      gemm_regular_only(A(i,0), B(0, j), C(i, j), alpha, beta);
      for (int k=1; k<A.dim[1]; k++) {
        gemm_regular_only(A(i, k), B(k, j), C(i, j), alpha, 1);
      }
    }
  }
}

define_method(
  void, gemm_regular_only_omm,
  (
    [[maybe_unused]] const LowRankShared& A, [[maybe_unused]] const Dense& B,
    Dense& C,
    [[maybe_unused]] double alpha, double beta
  )
) {
  // Only apply beta
  C *= beta;
}

define_method(
  void, gemm_regular_only_omm,
  (
    const Dense& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  gemm(A, B, C, alpha, beta);
}

define_method(
  void, gemm_regular_only_omm,
  (
    const Node& A, const Node& B, Node& C,
    [[maybe_unused]] double alpha, [[maybe_unused]] double beta
  )
) {
  std::cerr << "gemm_regular_only(";
  std::cerr << A.type() << "," << B.type() << "," << C.type();
  std::cerr << ") undefined." << std::endl;
  abort();
}

declare_method(
  bool, gemm_shared_only_omm,
  (
    virtual_<const Node&>, virtual_<const Node&>, virtual_<Node&>,
    double, double
  )
);

bool gemm_shared_only(
  const Node& A, const Node& B, Node& C, double alpha, double beta
) {
  return gemm_shared_only_omm(A, B, C, alpha, beta);
}

define_method(
  bool, gemm_shared_only_omm,
  (
    const LowRankShared& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  gemm(A.S, B, C, alpha, beta);
  return true;
}

define_method(
  bool, gemm_shared_only_omm,
  (
    [[maybe_unused]] const Dense& A, [[maybe_unused]] const Node& B,
    [[maybe_unused]] Node& C,
    [[maybe_unused]] double alpha, [[maybe_unused]] double beta
  )
) {
  // Do nothing
  return false;
}

define_method(
  bool, gemm_shared_only_omm,
  (
    [[maybe_unused]] const UniformHierarchical& A, [[maybe_unused]] const Node& B,
    [[maybe_unused]] Node& C,
    [[maybe_unused]] double alpha, [[maybe_unused]] double beta
  )
) {
  // Do nothing
  return false;
}

define_method(
  bool, gemm_shared_only_omm,
  (
    const Node& A, const Node& B, Node& C,
    [[maybe_unused]] double alpha, [[maybe_unused]] double beta
  )
) {
  std::cerr << "gemm_shared_only(";
  std::cerr << A.type() << "," << B.type() << "," << C.type();
  std::cerr << ") undefined." << std::endl;
  abort();
}

define_method(
  void, gemm_omm,
  (
    const UniformHierarchical& A, const Dense& B, Dense& C,
    double alpha, double beta
  )
) {
  C *= beta;
  NoCopySplit BH(B, A.dim[1], 1);
  NoCopySplit CH(C, A.dim[0], 1);
  // This function causes the recursion
  gemm_regular_only(A, BH, CH, alpha, 1);
  Hierarchical RowBasisB(1, A.dim[1]);
  for (int k=0; k<A.dim[1]; k++) {
    // TODO Need max rank here? Case for differing ranks not dealt with!
    // Find more elegant way to initialize (SplitDenseReference class?)
    RowBasisB[k] = Dense(get_n_rows(A.get_row_basis(0)), B.dim[1]);
  }
  // Loop over columns of output
  // Put together shared RowBasis and column of B once
  // Use result multiple times (faster) to get column of C
  for (int j=0; j<CH.dim[1]; j++) {
    // TODO Create RowColBasis class and define interactions for it.
    // The following loop really should be handled in those interactions.
    // This loop is main reason for speed-up: multiplication with BH only here,
    // rest is smaller stuff with rank as one of dimensios
    for (int k=0; k<A.dim[1]; k++) {
      gemm(A.get_row_basis(k), BH(k, j), RowBasisB[k], 1, 0);
    }
    for (int i=0; i<CH.dim[0]; i++) {
      Hierarchical SRowBasisB(1, RowBasisB.dim[1]);
      for (int k=0; k<A.dim[1]; k++) {
        SRowBasisB[k] = Dense(get_n_cols(A.get_col_basis(i)), get_n_cols(RowBasisB[k]));
        // Returns whether an operations took place (false when Dense/UH)
        bool shared = gemm_shared_only(A(i, k), RowBasisB[k], SRowBasisB[k], 1, 0);
        if (shared) {
          gemm(A.get_col_basis(i), SRowBasisB[k], CH(i, j), alpha, 1);
        }
      }
    }
  }
}

// Fallback default, abort with error message
define_method(
  void, gemm_omm,
  (
    const Node& A, const Node& B, Node& C,
    [[maybe_unused]] double alpha, [[maybe_unused]] double beta
  )
) {
  std::cerr << "gemm(";
  std::cerr << A.type() << "," << B.type() << "," << C.type();
  std::cerr << ") undefined." << std::endl;
  abort();
}

} // namespace hicma
