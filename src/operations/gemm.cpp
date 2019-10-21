#include "hicma/operations/gemm.h"

#include "hicma/node.h"
#include "hicma/node_proxy.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/util/timer.h"
#include "hicma/gpu_batch/batch.h"

#include <cassert>
#include <iostream>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

MULTI_METHOD(
  gemm_omm, void,
  const virtual_<Node>&,
  const virtual_<Node>&,
  virtual_<Node>&,
  const double,
  const double
);

void gemm(
  const NodeProxy& A, const NodeProxy& B, NodeProxy& C,
  const double alpha, const double beta
) {
  gemm(*A.ptr.get(), *B.ptr.get(), *C.ptr.get(), alpha, beta);
}
void gemm(
  const NodeProxy& A, const NodeProxy& B, Node& C,
  const double alpha, const double beta
) {
  gemm(*A.ptr.get(), *B.ptr.get(), C, alpha, beta);
}
void gemm(
  const NodeProxy& A, const Node& B, NodeProxy& C,
  const double alpha, const double beta
) {
  gemm(*A.ptr.get(), B, *C.ptr.get(), alpha, beta);
}
void gemm(
  const NodeProxy& A, const Node& B, Node& C,
  const double alpha, const double beta
) {
  gemm(*A.ptr.get(), B, C, alpha, beta);
}
void gemm(
  const Node& A, const NodeProxy& B, NodeProxy& C,
  const double alpha, const double beta
) {
  gemm(A, *B.ptr.get(), *C.ptr.get(), alpha, beta);
}
void gemm(
  const Node& A, const NodeProxy& B, Node& C,
  const double alpha, const double beta
) {
  gemm(A, *B.ptr.get(), C, alpha, beta);
}
void gemm(
  const Node& A, const Node& B, NodeProxy& C,
  const double alpha, const double beta
) {
  gemm(A, B, *C.ptr.get(), alpha, beta);
}

void gemm(
  const Node& A, const Node& B, Node& C,
  const double alpha, const double beta
) {
  gemm_omm(A, B, C, alpha, beta);
}
void gemm(
  const Dense& A, const Dense& B, Dense& C,
  const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
  const double& alpha, const double& beta
) {
  start("-DGEMM");
  if (B.dim[1] == 1) {
    cblas_dgemv(
      CblasRowMajor,
      CblasNoTrans,
      A.dim[0], A.dim[1],
      alpha,
      &A[0], A.dim[1],
      &B[0], 1,
      beta,
      &C.data[0], 1
    );
  }
  else {
    int k = TransA == CblasNoTrans ? A.dim[1] : A.dim[0];
    cblas_dgemm(
      CblasRowMajor,
      TransA, TransB,
      C.dim[0], C.dim[1], k,
      alpha,
      &A[0], A.dim[1],
      &B[0], B.dim[1],
      beta,
      &C.data[0], C.dim[1]
    );
  }
  stop("-DGEMM",false);
}

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Dense& A, const LowRank& B, LowRank& C,
  const double& alpha, const double& beta
) {
  LowRank AxBU(B);
  gemm(A, B.U, AxBU.U, alpha, 0);
  C.S *= beta;
  C += AxBU;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const LowRank& A, const Dense& B, LowRank& C,
  const double& alpha, const double& beta
) {
  LowRank AVxB(A);
  gemm(A.V, B, AVxB.V, alpha, 0);
  C.S *= beta;
  C += AVxB;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const LowRank& A, const LowRank& B, LowRank& C,
  const double& alpha, const double& beta
) {
  assert(A.rank == B.rank);
  LowRank AxB(A.dim[0], B.dim[1], A.rank);
  AxB.U = A.U;
  AxB.V = B.V;
  Dense VxU(A.rank, B.rank);
  gemm(A.V, B.U, VxU, 1, 0);
  Dense SxVxU(A.rank, B.rank);
  gemm(A.S, VxU, SxVxU, 1, 0);
  gemm(SxVxU, B.S, AxB.S, alpha, 0);
  C.S *= beta;
  C += AxB;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Dense& A, const Dense& B, LowRank& C,
  const double& alpha, const double& beta
) {
  assert(C.dim[0] == A.dim[0]);
  assert(A.dim[1] == B.dim[0]);
  assert(C.dim[1] == B.dim[1]);
  Dense AB(C.dim[0], C.dim[1]);
  gemm(A, B, AB, alpha, 0);
  C.S *= beta;
  C += LowRank(AB, C.rank);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Hierarchical& A, const LowRank& B, LowRank& C,
  const double& alpha, const double& beta
) {
  Hierarchical BH(B, A.dim[1], 1);
  Hierarchical CH(C, A.dim[0], 1);
  gemm(A, BH, CH, alpha, beta);
  // NOTE: This is likely inefficient!!
  // Make LowRank(Hierarchical) constructor?
  C = LowRank(Dense(CH), C.rank);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const LowRank& A, const Hierarchical& B, LowRank& C,
  const double& alpha, const double& beta
) {
  Hierarchical AH(A, 1, B.dim[0]);
  Hierarchical CH(C, 1, B.dim[1]);
  gemm(AH, B, CH, alpha, beta);
  // NOTE: This is likely inefficient!!
  // Make LowRank(Hierarchical) constructor?
  C = LowRank(Dense(CH), C.rank);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Hierarchical& A, const Hierarchical& B, LowRank& C,
  const double& alpha, const double& beta
) {
  /*
    Making a Hierarchical out of this might be better
    But LowRank(Hierarchical, rank) constructor is needed
    Hierarchical C(*this);
      gemm(A, B, C, alpha, beta);
    *this = LowRank(C, rank);
  */
  Dense CD(C);
  gemm(A, B, CD, alpha, beta);
  C = LowRank(CD, C.rank);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Dense& A, const Dense& B, Dense& C,
  const double& alpha, const double& beta
) {
  assert(C.dim[0] == A.dim[0]);
  assert(A.dim[1] == B.dim[0]);
  assert(C.dim[1] == B.dim[1]);
  if (alpha == 1 && beta == 1) {
    gemm_push(A, B, C);
  }
  else {
    gemm(A, B, C, CblasNoTrans, CblasNoTrans, alpha, beta);
  }
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Dense& A, const LowRank& B, Dense& C,
  const double& alpha, const double& beta
) {
  Dense AxU(C.dim[0], B.rank);
  gemm(A, B.U, AxU, 1, 0);
  Dense AxUxS(C.dim[0], B.rank);
  gemm(AxU, B.S, AxUxS, 1, 0);
  gemm(AxUxS, B.V, C, alpha, beta);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Dense& A, const Hierarchical& B, Dense& C,
  const double& alpha, const double& beta
) {
  Hierarchical CH(C, 1, B.dim[1]);
  gemm(A, B, CH, alpha, beta);
  C = Dense(CH);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const LowRank& A, const Dense& B, Dense& C,
  const double& alpha, const double& beta
) {
  Dense VxB(A.rank, B.dim[1]);
  gemm(A.V, B, VxB, 1, 0);
  Dense SxVxB(A.rank, B.dim[1]);
  gemm(A.S, VxB, SxVxB, 1, 0);
  gemm(A.U, SxVxB, C, alpha, beta);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const LowRank& A, const LowRank& B, Dense& C,
  const double& alpha, const double& beta
) {
  Dense VxU(A.rank, B.rank);
  gemm(A.V, B.U, VxU, 1, 0);
  Dense SxVxU(A.rank, B.rank);
  gemm(A.S, VxU, SxVxU, 1, 0);
  Dense SxVxUxS(A.rank, B.rank);
  gemm(SxVxU, B.S, SxVxUxS, 1, 0);
  Dense UxSxVxUxS(A.dim[0], B.rank);
  gemm(A.U, SxVxUxS, UxSxVxUxS, 1, 0);
  gemm(UxSxVxUxS, B.V, C, alpha, beta);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const LowRank& A, const Hierarchical& B, Dense& C,
  const double& alpha, const double& beta
) {
  Hierarchical CH(C, 1, B.dim[1]);
  gemm(A, B, CH, alpha, beta);
  C = Dense(CH);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Hierarchical& A, const Dense& B, Dense& C,
  const double& alpha, const double& beta
) {
  Hierarchical CH(C, A.dim[0], 1);
  gemm(A, B, CH, alpha, beta);
  C = Dense(CH);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Hierarchical& A, const LowRank& B, Dense& C,
  const double& alpha, const double& beta
) {
  Hierarchical CH(C, A.dim[0], 1);
  gemm(A, B, CH, alpha, beta);
  C = Dense(CH);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Hierarchical& A, const Hierarchical& B, Dense& C,
  const double& alpha, const double& beta
) {
  Hierarchical CH(C, A.dim[0], B.dim[1]);
  gemm(A, B, CH, alpha, beta);
  C = Dense(CH);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Dense& A, const Dense& B, Hierarchical& C,
  const double& alpha, const double& beta
) {
  assert(A.dim[1] == B.dim[0]);
  const Hierarchical& AH = Hierarchical(A, C.dim[0], 1);
  const Hierarchical& BH = Hierarchical(B, 1, C.dim[1]);
  gemm(AH, BH, C, alpha, beta);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Dense& A, const Hierarchical& B, Hierarchical& C,
  const double& alpha, const double& beta
) {
  const Hierarchical& AH = Hierarchical(A, C.dim[0], B.dim[0]);
  gemm(AH, B, C, alpha, beta);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const LowRank& A, const LowRank& B, Hierarchical& C,
  const double& alpha, const double& beta
) {
  const Hierarchical& AH = Hierarchical(A, C.dim[0], C.dim[0]);
  const Hierarchical& BH = Hierarchical(B, C.dim[1], C.dim[1]);
  gemm(AH, BH, C, alpha, beta);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const LowRank& A, const Hierarchical& B, Hierarchical& C,
  const double& alpha, const double& beta
) {
  const Hierarchical& AH = Hierarchical(A, C.dim[0], B.dim[0]);
  gemm(AH, B, C, alpha, beta);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Hierarchical& A, const Dense& B, Hierarchical& C,
  const double& alpha, const double& beta
) {
  const Hierarchical& BH = Hierarchical(B, A.dim[1], C.dim[1]);
  gemm(A, BH, C, alpha, beta);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Hierarchical& A, const LowRank& B, Hierarchical& C,
  const double& alpha, const double& beta
) {
  const Hierarchical& BH = Hierarchical(B, A.dim[1], C.dim[1]);
  gemm(A, BH, C, alpha, beta);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Hierarchical& A, const Hierarchical& B, Hierarchical& C,
  const double& alpha, const double& beta
) {
  assert(C.dim[0] == A.dim[0]);
  assert(C.dim[1] == B.dim[1]);
  assert(A.dim[1] == B.dim[0]);
  for (int i=0; i<C.dim[0]; i++) {
    for (int j=0; j<C.dim[1]; j++) {
      for (int k=0; k<A.dim[1]; k++) {
        gemm(A(i,k), B(k,j), C(i,j), alpha, beta);
      }
    }
  }
} END_SPECIALIZATION;

// Fallback default, abort with error message
BEGIN_SPECIALIZATION(
  gemm_omm, void,
  const Node& A, const Node& B, Node& C,
  const double& alpha, const double& beta
) {
  std::cerr << "gemm(";
  std::cerr << A.type() << "," << B.type() << "," << C.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
