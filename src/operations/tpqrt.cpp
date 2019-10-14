#include "hicma/operations/tpqrt.h"

#include "hicma/node.h"
#include "hicma/node_proxy.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/operations/gemm.h"
#include "hicma/operations/tpmqrt.h"

#include <iostream>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

void tpqrt(
  NodeProxy& A, NodeProxy& B, NodeProxy& T
) {
  tpqrt(*A.ptr, *B.ptr, *T.ptr);
}
void tpqrt(
  NodeProxy& A, NodeProxy& B, Node& T
) {
  tpqrt(*A.ptr, *B.ptr, T);
}
void tpqrt(
  NodeProxy& A, Node& B, NodeProxy& T
) {
  tpqrt(*A.ptr, B, *T.ptr);
}
void tpqrt(
  NodeProxy& A, Node& B, Node& T
) {
  tpqrt(*A.ptr, B, T);
}
void tpqrt(
  Node& A, NodeProxy& B, NodeProxy& T
) {
  tpqrt(A, *B.ptr, *T.ptr);
}
void tpqrt(
  Node& A, NodeProxy& B, Node& T
) {
  tpqrt(A, *B.ptr, T);
}
void tpqrt(
  Node& A, Node& B, NodeProxy& T
) {
  tpqrt(A, B, *T.ptr);
}

void tpqrt(
  Node& A, Node& B, Node& T
) {
  tpqrt_omm(A, B, T);
}

BEGIN_SPECIALIZATION(
  tpqrt_omm, void,
  Dense& A, Dense& B, Dense& T
) {
  LAPACKE_dtpqrt2(
    LAPACK_ROW_MAJOR,
    B.dim[0], B.dim[1], 0,
    &A[0], A.dim[1],
    &B[0], B.dim[1],
    &T[0], T.dim[1]
  );
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpqrt_omm, void,
  Dense& A, LowRank& B, Dense& T
) {
  Dense BV_copy(B.V);
  gemm(B.S, BV_copy, B.V, CblasNoTrans, CblasNoTrans, 1, 0);
  for(int i = 0; i < std::min(B.S.dim[0], B.S.dim[1]); i++) B.S(i, i) = 1.0;
  tpqrt(A, B.V, T);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpqrt_omm, void,
  Hierarchical& A, Hierarchical& B, Hierarchical& T
) {
  for(int i = 0; i < B.dim[0]; i++) {
    for(int j = 0; j < B.dim[1]; j++) {
      tpqrt(A(j, j), B(i, j), T(i, j));
      for(int k = j+1; k < B.dim[1]; k++) {
        tpmqrt(B(i, j), T(i, j), A(j, k), B(i, k), true);
      }
    }
  }
} END_SPECIALIZATION;

// Fallback default, abort with error message
BEGIN_SPECIALIZATION(
  tpqrt_omm, void,
  Node& A, Node& B, Node& T
) {
  std::cerr << "tpqrt(";
  std::cerr << A.type() << "," << B.type() << "," << T.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma