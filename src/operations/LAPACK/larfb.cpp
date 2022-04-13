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

// single precision
define_method(
  void, larfb_omm,
  (const Dense<float>& V, const Dense<float>& T, Dense<float>& C, bool trans)
) {
  LAPACKE_slarfb(
    LAPACK_ROW_MAJOR,
    'L', (trans ? 'T' : 'N'), 'F', 'C',
    C.dim[0], C.dim[1], T.dim[1],
    &V, V.stride,
    &T, T.stride,
    &C, C.stride
  );
}

// double precision
define_method(
  void, larfb_omm,
  (const Dense<double>& V, const Dense<double>& T, Dense<double>& C, bool trans)
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

template<typename T>
void larfb_h_h_d(const Hierarchical<T>& V, const Hierarchical<T>& W, Dense<T>& C, bool trans) {
  Hierarchical<T> CH = split<T>(C, V.dim[0], V.dim[1], true);
  larfb(V, W, CH, trans);
  C = Dense<T>(CH);
}

define_method(
  void, larfb_omm,
  (const Hierarchical<float>& V, const Hierarchical<float>& T, Dense<float>& C, bool trans)
) {
  larfb_h_h_d(V, T, C, trans);
}

define_method(
  void, larfb_omm,
  (const Hierarchical<double>& V, const Hierarchical<double>& T, Dense<double>& C, bool trans)
) {
  larfb_h_h_d(V, T, C, trans);
}

define_method(
  void, larfb_omm,
  (const Dense<float>& V, const Dense<float>& T, LowRank<float>& C, bool trans)
) {
  larfb(V, T, C.U, trans);
}

define_method(
  void, larfb_omm,
  (const Dense<double>& V, const Dense<double>& T, LowRank<double>& C, bool trans)
) {
  larfb(V, T, C.U, trans);
}

template<typename T>
void larfb_d_d_h(const Dense<T>& V, const Dense<T>& W, Hierarchical<T>& C, bool trans) {
  Dense<T> V_lower_tri(V);
  for(int64_t i = 0; i < V_lower_tri.dim[0]; i++) {
    for(int64_t j = i; j < V_lower_tri.dim[1]; j++) {
      if(i == j) V_lower_tri(i, j) = 1.0;
      else V_lower_tri(i, j) = 0.0;
    }
  }
  Dense<T> VW(V_lower_tri);
  trmm(W, VW, 'r', 'u', trans ? 't' : 'n', 'n', 1);
  Dense<T> VWVt(VW.dim[0], V_lower_tri.dim[0]);
  gemm(VW, V_lower_tri, VWVt, 1, 0, false, true);
  Hierarchical<T> C_copy(C);
  gemm(VWVt, C_copy, C, -1, 1);
}

define_method(
  void, larfb_omm,
  (const Dense<float>& V, const Dense<float>& T, Hierarchical<float>& C, bool trans)
) {
  larfb_d_d_h(V, T, C, trans);
}

define_method(
  void, larfb_omm,
  (const Dense<double>& V, const Dense<double>& T, Hierarchical<double>& C, bool trans)
) {
  larfb_d_d_h(V, T, C, trans);
}

template<typename T>
void larfb_h_h_h(const Hierarchical<T>& V, const Hierarchical<T>& W, Hierarchical<T>& C, bool trans) {
  if(trans) {
    for(int64_t k = 0; k < C.dim[1]; k++) {
      for(int64_t j = k; j < C.dim[1]; j++) {
        larfb(V(k, k), W(k, k), C(k, j), trans);
      }
      for(int64_t i = k+1; i < C.dim[0]; i++) {
        for(int64_t j = k; j < C.dim[1]; j++) {
          tpmqrt(V(i, k), W(i, k), C(k, j), C(i, j), trans);
        }
      }
    }
  }
  else {
    for(int64_t k = C.dim[1]-1; k >= 0; k--) {
      for(int64_t i = C.dim[0]-1; i > k; i--) {
        for(int64_t j = k; j < C.dim[1]; j++) {
          tpmqrt(V(i, k), W(i, k), C(k, j), C(i, j), trans);
        }
      }
      for(int64_t j = k; j < C.dim[1]; j++) {
        larfb(V(k, k), W(k, k), C(k, j), trans);
      }
    }
  }
}

define_method(
  void, larfb_omm,
  (const Hierarchical<float>& V, const Hierarchical<float>& T, Hierarchical<float>& C, bool trans)
) {
  larfb_h_h_h(V, T, C, trans);
}

define_method(
  void, larfb_omm,
  (const Hierarchical<double>& V, const Hierarchical<double>& T, Hierarchical<double>& C, bool trans)
) {
  larfb_h_h_h(V, T, C, trans);
}

// Fallback default, abort with error message
define_method(
  void, larfb_omm, (const Matrix& V, const Matrix& T, Matrix& C, bool)
) {
  omm_error_handler("larfb", {V, T, C}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
