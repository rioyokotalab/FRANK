#include "hicma/operations/larfb.h"

#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/operations/gemm.h"
#include "hicma/operations/tpmqrt.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

MULTI_METHOD(
  larfb_omm, void,
  const virtual_<Node>&,
  const virtual_<Node>&,
  virtual_<Node>&,
  const bool
);

void larfb(
  const NodeProxy& V, const NodeProxy& T, NodeProxy& C,
  const bool trans
) {
  larfb(*V.ptr, *T.ptr, *C.ptr, trans);
}
void larfb(
  const NodeProxy& V, const NodeProxy& T, Node& C,
  const bool trans
) {
  larfb(*V.ptr, *T.ptr, C, trans);
}
void larfb(
  const NodeProxy& V, const Node& T, NodeProxy& C,
  const bool trans
) {
  larfb(*V.ptr, T, *C.ptr, trans);
}
void larfb(
  const NodeProxy& V, const Node& T, Node& C,
  const bool trans
) {
  larfb(*V.ptr, T, C, trans);
}
void larfb(
  const Node& V, const NodeProxy& T, NodeProxy& C,
  const bool trans
) {
  larfb(V, *T.ptr, *C.ptr, trans);
}
void larfb(
  const Node& V, const NodeProxy& T, Node& C,
  const bool trans
) {
  larfb(V, *T.ptr, C, trans);
}
void larfb(
  const Node& V, const Node& T, NodeProxy& C,
  const bool trans
) {
  larfb(V, T, *C.ptr, trans);
}

void larfb(
  const Node& V, const Node& T, Node& C,
  const bool trans
) {
  larfb_omm(V, T, C, trans);
}

BEGIN_SPECIALIZATION(
  larfb_omm, void,
  const Dense& V, const Dense& T, Dense& C,
  const bool trans
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
  const bool trans
) {
  Hierarchical CH(C, V.dim[0], V.dim[1]);
  larfb(V, T, CH, trans);
  C = Dense(CH);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  larfb_omm, void,
  const Dense& V, const Dense& T, LowRank& C,
  const bool trans
) {
  larfb(V, T, C.U, trans);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  larfb_omm, void,
  const Dense& V, const Dense& T, Hierarchical& C,
  const bool trans
) {
  Dense V_lower_tri(V);
  for(int i = 0; i < V_lower_tri.dim[0]; i++) {
    for(int j = i; j < V_lower_tri.dim[1]; j++) {
      if(i == j) V_lower_tri(i, j) = 1.0;
      else V_lower_tri(i, j) = 0.0;
    }
  }
  Dense VT(V_lower_tri.dim[0], T.dim[1]);
  gemm(V_lower_tri, T, VT, CblasNoTrans, trans ? CblasTrans : CblasNoTrans, 1, 1);
  Dense YTYt(VT.dim[0], V_lower_tri.dim[0]);
  gemm(VT, V_lower_tri, YTYt, CblasNoTrans, CblasTrans, 1, 1);
  Hierarchical C_copy(C);
  gemm(YTYt, C_copy, C, -1, 1);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  larfb_omm, void,
  const Hierarchical& V, const Hierarchical& T, Hierarchical& C,
  const bool trans
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
  const bool trans
) {
  std::cerr << "larfb(";
  std::cerr << V.type() << "," << T.type() << "," << C.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
