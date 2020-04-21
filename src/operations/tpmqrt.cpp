#include "hicma/operations/tpmqrt.h"

#include "hicma/node.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations/gemm.h"
#include "hicma/operations/trmm.h"

#include <iostream>
#include <vector>

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
    B.dim[0], B.dim[1], T.dim[1], 0,
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
  std::vector<double> x;
  Dense C(A);
  LowRank Vt(V); Vt.transpose();
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(Dense(identity, x, C.dim[0], C.dim[0]), C, A, -1, 1); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Dense& V, const Dense& T, LowRank& A, Dense& B,
  const bool trans
) {
  std::vector<double> x;
  Dense C(A);
  gemm(V, B, C, CblasTrans, CblasNoTrans, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(Dense(identity, x, C.dim[0], C.dim[0]), C, A, -1, 1); //A = A - I*C //Recompression
  gemm(V, C, B, -1, 1); //B = B - Y*C
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const LowRank& V, const Dense& T, LowRank& A, Dense& B,
  const bool trans
) {
  std::vector<double> x;
  LowRank C(A);
  LowRank Vt(V);
  Vt.transpose();
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(Dense(identity, x, C.dim[0], C.dim[0]), C, A, -1, 1); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Dense& V, const Dense& T, Hierarchical& A, Dense& B,
  const bool trans
) {
  Dense Vt(V);
  Vt.transpose();
  Dense T_upper_tri(T);
  for(int i=0; i<T_upper_tri.dim[0]; i++)
    for(int j=0; j<i; j++)
      T_upper_tri(i, j) = 0.0;
  Hierarchical AH(A);
  gemm(Vt, B, AH, 1, 1); // AH = A + Vt*B
  if(trans) T_upper_tri.transpose();
  gemm(T_upper_tri, AH, A, -1, 1); // A = A - (T or Tt)*AH
  Dense VTt(V.dim[0], T_upper_tri.dim[1]);
  gemm(V, T_upper_tri, VTt, 1, 0);
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
  std::vector<double> x;
  Dense C(A);
  Dense Vt(V);
  Vt.transpose();
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(Dense(identity, x, C.dim[0], C.dim[0]), C, A, -1, 1); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const LowRank& V, const Dense& T, Dense& A, LowRank& B,
  const bool trans
) {
  std::vector<double> x;
  Dense C(A);
  LowRank Vt(V);
  Vt.transpose();
  gemm(Vt, B, C, 1, 1); //C = A + Y^t * B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(Dense(identity, x, C.dim[0], C.dim[0]), C, A, -1, 1); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Dense& V, const Dense& T, LowRank& A, LowRank& B,
  const bool trans
) {
  std::vector<double> x;
  LowRank C(A);
  Dense Vt(V);
  Vt.transpose();
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(Dense(identity, x, C.dim[0], C.dim[0]), C, A, -1, 1); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const LowRank& V, const Dense& T, LowRank& A, LowRank& B,
  const bool trans
) {
  std::vector<double> x;
  LowRank C(A);
  LowRank Vt(V);
  Vt.transpose();
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(Dense(identity, x, C.dim[0], C.dim[0]), C, A, -1, 1); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  tpmqrt_omm, void,
  const Dense& V, const Dense& T, Dense& A, Hierarchical& B,
  const bool trans
) {
  Dense C(A);
  Dense Vt(V);
  Vt.transpose();
  Dense T_upper_tri(T);
  for(int i=0; i<T_upper_tri.dim[0]; i++)
    for(int j=0; j<i; j++)
      T_upper_tri(i, j) = 0.0;
  gemm(Vt, B, C, 1, 1); // C = A + Y^t*B
  if(trans) T_upper_tri.transpose();
  gemm(T_upper_tri, C, A, -1, 1); // A = A - (T or Tt)*C
  Dense VTt(V.dim[0], T_upper_tri.dim[1]);
  gemm(V, T_upper_tri, VTt, 1, 1);
  gemm(VTt, C, B, -1, 1); // B = B - V*(T or Tt)*C
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

