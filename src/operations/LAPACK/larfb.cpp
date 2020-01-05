#include "hicma/operations/LAPACK/larfb.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/BLAS/trmm.h"
#include "hicma/operations/LAPACK/tpmqrt.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

void larfb(
  const Node& V, const Node& T, Node& C,
  bool trans
) {
  larfb_omm(V, T, C, trans);
}

BEGIN_SPECIALIZATION(
  larfb_omm, void,
  const Dense& V, const Dense& T, Dense& C,
  bool trans
) {
  LAPACKE_dlarfb(
    LAPACK_ROW_MAJOR,
    'L', (trans ? 'T' : 'N'), 'F', 'C',
    C.dim[0], C.dim[1], T.dim[1],
    &V[0], V.dim[1],
    &T[0], T.dim[1],
    &C[0], C.dim[1]
  );
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  larfb_omm, void,
  const Hierarchical& V, const Hierarchical& T, Dense& C,
  bool trans
) {
  Hierarchical CH(C, V.dim[0], V.dim[1]);
  larfb(V, T, CH, trans);
  C = Dense(CH);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  larfb_omm, void,
  const Dense& V, const Dense& T, LowRank& C,
  bool trans
) {
  larfb(V, T, C.U, trans);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  larfb_omm, void,
  const Dense& V, const Dense& T, Hierarchical& C,
  bool trans
) {
  Dense V_lower_tri(V);
  for(int i = 0; i < V_lower_tri.dim[0]; i++) {
    for(int j = i; j < V_lower_tri.dim[1]; j++) {
      if(i == j) V_lower_tri(i, j) = 1.0;
      else V_lower_tri(i, j) = 0.0;
    }
  }
  Dense VT(V_lower_tri);
  trmm(T, VT, 'r', 'u', trans ? 't' : 'n', 'n', 1);
  Dense VTVt(VT.dim[0], V_lower_tri.dim[0]);
  gemm(VT, V_lower_tri, VTVt, false, true, 1, 0);
  Hierarchical C_copy(C);
  gemm(VTVt, C_copy, C, -1, 1);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  larfb_omm, void,
  const Hierarchical& V, const Hierarchical& T, Hierarchical& C,
  bool trans
) {
  if(trans) {
    for(int k = 0; k < C.dim[1]; k++) {
      for(int j = k; j < C.dim[1]; j++) {
        larfb(V(k, k), T(k, k), C(k, j), trans);
      }
      for(int i = k+1; i < C.dim[0]; i++) {
        for(int j = k; j < C.dim[1]; j++) {
          tpmqrt(V(i, k), T(i, k), C(k, j), C(i, j), trans);
        }
      }
    }
  }
  else {
    for(int k = C.dim[1]-1; k >= 0; k--) {
      for(int i = C.dim[0]-1; i > k; i--) {
        for(int j = k; j < C.dim[1]; j++) {
          tpmqrt(V(i, k), T(i, k), C(k, j), C(i, j), trans);
        }
      }
      for(int j = k; j < C.dim[1]; j++) {
        larfb(V(k, k), T(k, k), C(k, j), trans);
      }
    }
  }
} END_SPECIALIZATION;

// Fallback default, abort with error message
BEGIN_SPECIALIZATION(
  larfb_omm, void,
  const Node& V, const Node& T, Node& C,
  bool trans
) {
  std::cerr << "larfb(";
  std::cerr << V.type() << "," << T.type() << "," << C.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
