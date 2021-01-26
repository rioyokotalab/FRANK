#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"

#include <cstdint>
#include <cstdlib>


namespace hicma
{

void larfb(const Matrix& V, const Matrix& T, Matrix& C, bool trans) {
  larfb_omm(V, T, C, trans);
}

define_method(
  void, larfb_omm,
  (const Dense& V, const Dense& T, Dense& C, bool trans)
) {
  LAPACKE_dlarfb(
    LAPACK_ROW_MAJOR,
    'L', (trans ? 'T' : 'N'), 'F', 'C',
    C.dim[0], C.dim[1], T.dim[1],
    &V, V.stride,
    &T, T.stride,
    &C, C.stride
  );
}

define_method(
  void, larfb_omm,
  (const Hierarchical& V, const Hierarchical& T, Dense& C, bool trans)
) {
  Hierarchical CH = split(C, V.dim[0], V.dim[1], true);
  larfb(V, T, CH, trans);
  C = Dense(CH);
}

define_method(
  void, larfb_omm,
  (const Dense& V, const Dense& T, LowRank& C, bool trans)
) {
  larfb(V, T, C.U, trans);
}

define_method(
  void, larfb_omm,
  (const Dense& V, const Dense& T, Hierarchical& C, bool trans)
) {
  Dense V_lower_tri(V);
  for(int64_t i = 0; i < V_lower_tri.dim[0]; i++) {
    for(int64_t j = i; j < V_lower_tri.dim[1]; j++) {
      if(i == j) V_lower_tri(i, j) = 1.0;
      else V_lower_tri(i, j) = 0.0;
    }
  }
  Dense VT(V_lower_tri);
  trmm(T, VT, 'r', 'u', trans ? 't' : 'n', 'n', 1);
  Dense VTVt(VT.dim[0], V_lower_tri.dim[0]);
  gemm(VT, V_lower_tri, VTVt, 1, 0, false, true);
  Hierarchical C_copy(C);
  gemm(VTVt, C_copy, C, -1, 1);
}

define_method(
  void, larfb_omm,
  (const Hierarchical& V, const Hierarchical& T, Hierarchical& C, bool trans)
) {
  if(trans) {
    for(int64_t k = 0; k < C.dim[1]; k++) {
      for(int64_t j = k; j < C.dim[1]; j++) {
        larfb(V(k, k), T(k, k), C(k, j), trans);
      }
      for(int64_t i = k+1; i < C.dim[0]; i++) {
        for(int64_t j = k; j < C.dim[1]; j++) {
          tpmqrt(V(i, k), T(i, k), C(k, j), C(i, j), trans);
        }
      }
    }
  }
  else {
    for(int64_t k = C.dim[1]-1; k >= 0; k--) {
      for(int64_t i = C.dim[0]-1; i > k; i--) {
        for(int64_t j = k; j < C.dim[1]; j++) {
          tpmqrt(V(i, k), T(i, k), C(k, j), C(i, j), trans);
        }
      }
      for(int64_t j = k; j < C.dim[1]; j++) {
        larfb(V(k, k), T(k, k), C(k, j), trans);
      }
    }
  }
}

// Fallback default, abort with error message
define_method(
  void, larfb_omm, (const Matrix& V, const Matrix& T, Matrix& C, bool)
) {
  omm_error_handler("larfb", {V, T, C}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
