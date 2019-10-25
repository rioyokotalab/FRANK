#include "hicma/operations/tpmqrt.h"

#include "hicma/node.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/operations/gemm.h"

#include <iostream>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

void tpmqrt(
  const Node& V, const Node& T, Node& A, Node& B,
  const bool trans
) {
  tpmqrt_omm(V, T, A, B, trans);
}

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Dense& V, const Dense& T, Dense& A, Dense& B,
  const bool trans
) {
  LAPACKE_dtprfb(
    LAPACK_ROW_MAJOR,
    'L', (trans ? 'T': 'N'), 'F', 'C',
    A.dim[0], A.dim[1], V.dim[1], 0,
    &V[0], V.dim[1],
    &T[0], T.dim[1],
    &A[0], A.dim[1],
    &B[0], B.dim[1]
  );
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const LowRank& V, const Dense& T, Dense& A, Dense& B,
  const bool trans
) {
  Dense UV(V.U.dim[0], V.V.dim[1]);
  gemm(V.U, V.V, UV, 1, 0);
  tpmqrt(UV, T, A, B, trans);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Dense& V, const Dense& T, LowRank& A, Dense& B,
  const bool trans
) {
  Dense AD(A);
  tpmqrt(V, T, AD, B, trans);
  A = LowRank(AD, A.rank);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const LowRank& V, const Dense& T, LowRank& A, Dense& B,
  const bool trans
) {
  Dense UV(V.U.dim[0], V.V.dim[1]);
  gemm(V.U, V.V, UV, 1, 0);
  Dense AD(A);
  tpmqrt(UV, T, AD, B, trans);
  A = LowRank(AD, A.rank);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Dense& V, const Dense& T, Hierarchical& A, Dense& B,
  const bool trans
) {
  Dense Vt(V);
  Vt.transpose();
  Hierarchical AH(A);
  gemm(Vt, B, AH, 1, 1); // AH = A + Vt*B
  Dense Tt(T);
  if(trans) Tt.transpose();
  gemm(Tt, AH, A, -1, 1); // A = A - (T or Tt)*AH
  Dense VTt(V.dim[0], Tt.dim[1]);
  gemm(V, Tt, VTt, 1, 0);
  gemm(VTt, AH, B, -1, 1); // B = B - V*(T or Tt)*AH
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Hierarchical& V, const Hierarchical& T, Hierarchical& A, Dense& B,
  const bool trans
) {
  Hierarchical BH(B, A.dim[0], A.dim[1]);
  tpmqrt(V, T, A, BH, trans);
  B = Dense(BH);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Dense& V, const Dense& T, Dense& A, LowRank& B,
  const bool trans
) {
  Dense BD(B);
  tpmqrt(V, T, A, BD, trans);
  B = LowRank(BD, B.rank);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const LowRank& V, const Dense& T, Dense& A, LowRank& B,
  const bool trans
) {
  Dense UV(V.U.dim[0], V.V.dim[1]);
  gemm(V.U, V.V, UV, 1, 0);
  Dense BD(B);
  tpmqrt(UV, T, A, BD, trans);
  B = LowRank(BD, B.rank);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Dense& V, const Dense& T, LowRank& A, LowRank& B,
  const bool trans
) {
  Dense AD(A);
  Dense BD(B);
  tpmqrt(V, T, AD, BD, trans);
  A = LowRank(AD, A.rank);
  B = LowRank(BD, B.rank);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const LowRank& V, const Dense& T, LowRank& A, LowRank& B,
  const bool trans
) {
  Dense UV(V.U.dim[0], V.V.dim[1]);
  gemm(V.U, V.V, UV, 1, 0);
  Dense AD(A);
  Dense BD(B);
  tpmqrt(UV, T, AD, BD, trans);
  A = LowRank(AD, A.rank);
  B = LowRank(BD, B.rank);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Dense& V, const Dense& T, Dense& A, Hierarchical& B,
  const bool trans
) {
    Dense Vt(V);
    Vt.transpose();
    Dense A_copy(A);
    gemm(Vt, B, A_copy, 1, 1); // A_copy = A + Vt*B
    Dense Tt(T);
    if(trans) Tt.transpose();
    gemm(Tt, A_copy, A, -1, 1); // A = A - (T or Tt)*A_copy
    Dense VTt(V.dim[0], Tt.dim[1]);
    gemm(V, Tt, VTt, 1, 1);
    gemm(VTt, A_copy, B, -1, 1); // B = B - V*(T or Tt)*A_copy
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Hierarchical& V, const Hierarchical& T, Dense& A, Hierarchical& B,
  const bool trans
) {
  Hierarchical HA(A, B.dim[0], B.dim[1]);
  tpmqrt(V, T, HA, B, trans);
  A = Dense(HA);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Hierarchical& V, const Hierarchical& T, Hierarchical& A, Hierarchical& B,
  const bool trans
) {
  if(trans) {
    for(int i = 0; i < B.dim[0]; i++) {
      for(int j = 0; j < B.dim[1]; j++) {
        for(int k = 0; k < B.dim[1]; k++) {
          tpmqrt(V(i, j), T(i, j), A(j, k), B(i, k), trans);
        }
      }
    }
  }
  else {
    for(int i = B.dim[0]-1; i >= 0; i--) {
      for(int j = B.dim[1]-1; j >= 0; j--) {
        for(int k = B.dim[1]-1; k >= 0; k--) {
          tpmqrt(V(i, j), T(i, j), A(j, k), B(i, k), trans);
        }
      }
    }
  }
} END_SPECIALIZATION;

// Fallback default, abort with error message
BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Node& V, const Node& T, Node& A, Node& B,
  const bool trans
) {
  std::cerr << "tpmqrt(";
  std::cerr << V.type() << "," << T.type() << "," << A.type() << "," << B.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
