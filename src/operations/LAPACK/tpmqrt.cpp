#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/functions.h"
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
#include <vector>


namespace hicma
{

void tpmqrt(
  const Matrix& V, const Matrix& T, Matrix& A, Matrix& B, bool trans
) {
  tpmqrt_omm(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const Dense& V, const Dense& T, Dense& A, Dense& B, bool trans)
) {
  LAPACKE_dtprfb(
    LAPACK_ROW_MAJOR,
    'L', (trans ? 'T': 'N'), 'F', 'C',
    B.dim[0], B.dim[1], T.dim[1], 0,
    &V, V.stride,
    &T, T.stride,
    &A, A.stride,
    &B, B.stride
  );
}

define_method(
  void, tpmqrt_omm,
  (const LowRank& V, const Dense& T, Dense& A, Dense& B, bool trans)
) {
  Dense C(A);
  LowRank Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, std::vector<std::vector<double>>(), C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense& V, const Dense& T, LowRank& A, Dense& B, bool trans)
) {
  Dense C(A);
  gemm(V, B, C, true, false, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, std::vector<std::vector<double>>(), C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C //Recompression
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const LowRank& V, const Dense& T, LowRank& A, Dense& B, bool trans)
) {
  LowRank C(A);
  LowRank Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, std::vector<std::vector<double>>(), C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense& V, const Dense& T, Hierarchical& A, Dense& B, bool trans)
) {
  Dense Vt = transpose(V);
  Dense T_upper_tri(T);
  for(int64_t i=0; i<T_upper_tri.dim[0]; i++)
    for(int64_t j=0; j<i; j++)
      T_upper_tri(i, j) = 0.0;
  Hierarchical AH(A);
  gemm(Vt, B, AH, 1, 1); // AH = A + Vt*B
  if(trans) T_upper_tri = transpose(T_upper_tri);
  gemm(T_upper_tri, AH, A, -1, 1); // A = A - (T or Tt)*AH
  Dense VTt = gemm(V, T_upper_tri);
  gemm(VTt, AH, B, -1, 1); // B = B - V*(T or Tt)*AH
}

define_method(
  void, tpmqrt_omm,
  (
    const Hierarchical& V, const Hierarchical& T, Hierarchical& A, Dense& B,
    bool trans
  )
) {
  Hierarchical BH(B, A.dim[0], A.dim[1]);
  tpmqrt(V, T, A, BH, trans);
  B = Dense(BH);
}

define_method(
  void, tpmqrt_omm,
  (const Dense& V, const Dense& T, Dense& A, LowRank& B, bool trans)
) {
  Dense C(A);
  Dense Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, std::vector<std::vector<double>>(), C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const LowRank& V, const Dense& T, Dense& A, LowRank& B, bool trans)
) {
  Dense C(A);
  LowRank Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t * B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, std::vector<std::vector<double>>(), C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense& V, const Dense& T, LowRank& A, LowRank& B, bool trans)
) {
  LowRank C(A);
  Dense Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, std::vector<std::vector<double>>(), C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const LowRank& V, const Dense& T, LowRank& A, LowRank& B, bool trans)
) {
  LowRank C(A);
  LowRank Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, std::vector<std::vector<double>>(), C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense& V, const Dense& T, Dense& A, Hierarchical& B, bool trans)
) {
  Dense C(A);
  Dense Vt = transpose(V);
  Dense T_upper_tri(T);
  for(int64_t i=0; i<T_upper_tri.dim[0]; i++)
    for(int64_t j=0; j<i; j++)
      T_upper_tri(i, j) = 0.0;
  gemm(Vt, B, C, 1, 1); // C = A + Y^t*B
  if(trans) T_upper_tri = transpose(T_upper_tri);
  gemm(T_upper_tri, C, A, -1, 1); // A = A - (T or Tt)*C
  Dense VTt(V.dim[0], T_upper_tri.dim[1]);
  gemm(V, T_upper_tri, VTt, 1, 1);
  gemm(VTt, C, B, -1, 1); // B = B - V*(T or Tt)*C
}

define_method(
  void, tpmqrt_omm,
  (
    const Hierarchical& V, const Hierarchical& T, Dense& A, Hierarchical& B,
    bool trans
  )
) {
  Hierarchical HA(A, B.dim[0], B.dim[1]);
  tpmqrt(V, T, HA, B, trans);
  A = Dense(HA);
}

define_method(
  void, tpmqrt_omm,
  (
    const Hierarchical& V, const Hierarchical& T,
    Hierarchical& A, Hierarchical& B,
    bool trans
  )
) {
  if(trans) {
    for(int64_t i = 0; i < B.dim[0]; i++) {
      for(int64_t j = 0; j < B.dim[1]; j++) {
        for(int64_t k = 0; k < B.dim[1]; k++) {
          tpmqrt(V(i, j), T(i, j), A(j, k), B(i, k), trans);
        }
      }
    }
  }
  else {
    for(int64_t i = B.dim[0]-1; i >= 0; i--) {
      for(int64_t j = B.dim[1]-1; j >= 0; j--) {
        for(int64_t k = B.dim[1]-1; k >= 0; k--) {
          tpmqrt(V(i, j), T(i, j), A(j, k), B(i, k), trans);
        }
      }
    }
  }
}

// Fallback default, abort with error message
define_method(
  void, tpmqrt_omm,
  (
    const Matrix& V, const Matrix& T, Matrix& A, Matrix& B,
    [[maybe_unused]] bool trans
  )
) {
  omm_error_handler("tpmqrt", {V, T, A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma

