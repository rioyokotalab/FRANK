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

// single precision
define_method(
  void, tpmqrt_omm,
  (const Dense<float>& V, const Dense<float>& T, Dense<float>& A, Dense<float>& B, bool trans)
) {
  LAPACKE_stprfb(
    LAPACK_ROW_MAJOR,
    'L', (trans ? 'T': 'N'), 'F', 'C',
    B.dim[0], B.dim[1], T.dim[1], 0,
    &V, V.stride,
    &T, T.stride,
    &A, A.stride,
    &B, B.stride
  );
}

// double precision
define_method(
  void, tpmqrt_omm,
  (const Dense<double>& V, const Dense<double>& T, Dense<double>& A, Dense<double>& B, bool trans)
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

template<typename T>
void tpmqrt_lr_d_d_d(const LowRank<T>& V, const Dense<T>& S, Dense<T>& A, Dense<T>& B, bool trans) {
  Dense<T> C(A);
  LowRank<T> Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(S, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = S*C or S^t*C
  gemm(
    Dense<T>(identity, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const LowRank<float>& V, const Dense<float>& T, Dense<float>& A, Dense<float>& B, bool trans)
) {
  tpmqrt_lr_d_d_d(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const LowRank<double>& V, const Dense<double>& T, Dense<double>& A, Dense<double>& B, bool trans)
) {
  tpmqrt_lr_d_d_d(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_d_d_lr_d(const Dense<T>& V, const Dense<T>& S, LowRank<T>& A, Dense<T>& B, bool trans) {
  Dense<T> C(A);
  gemm(V, B, C, 1, 1, true, false); //C = A + Y^t*B
  trmm(S, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = S*C or S^t*C
  gemm(
    Dense<T>(identity, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C //Recompression
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense<float>& V, const Dense<float>& T, LowRank<float>& A, Dense<float>& B, bool trans)
) {
  tpmqrt_d_d_lr_d(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const Dense<double>& V, const Dense<double>& T, LowRank<double>& A, Dense<double>& B, bool trans)
) {
  tpmqrt_d_d_lr_d(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_lr_d_lr_d(const LowRank<T>& V, const Dense<T>& S, LowRank<T>& A, Dense<T>& B, bool trans) {
  LowRank<T> C(A);
  LowRank<T> Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(S, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = S*C or S^t*C
  gemm(
    Dense<T>(identity, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const LowRank<float>& V, const Dense<float>& T, LowRank<float>& A, Dense<float>& B, bool trans)
) {
  tpmqrt_lr_d_lr_d(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const LowRank<double>& V, const Dense<double>& T, LowRank<double>& A, Dense<double>& B, bool trans)
) {
  tpmqrt_lr_d_lr_d(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_d_d_h_d(const Dense<T>& V, const Dense<T>& S, Hierarchical<T>& A, Dense<T>& B, bool trans) {
  Dense<T> Vt = transpose(V);
  Dense<T> S_upper_tri(S);
  for(int64_t i=0; i<S_upper_tri.dim[0]; i++)
    for(int64_t j=0; j<i; j++)
      S_upper_tri(i, j) = 0.0;
  Hierarchical<T> AH(A);
  gemm(Vt, B, AH, 1, 1); // AH = A + Vt*B
  if(trans) S_upper_tri = transpose(S_upper_tri);
  gemm(S_upper_tri, AH, A, -1, 1); // A = A - (S or St)*AH
  Dense<T> VSt = gemm(V, S_upper_tri);
  gemm(VSt, AH, B, -1, 1); // B = B - V*(S or St)*AH
}

define_method(
  void, tpmqrt_omm,
  (const Dense<float>& V, const Dense<float>& T, Hierarchical<float>& A, Dense<float>& B, bool trans)
) {
  tpmqrt_d_d_h_d(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const Dense<double>& V, const Dense<double>& T, Hierarchical<double>& A, Dense<double>& B, bool trans)
) {
  tpmqrt_d_d_h_d(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_h_h_h_d(const Hierarchical<T>& V, const Hierarchical<T>& S, Hierarchical<T>& A, Dense<T>& B, bool trans) {
  Hierarchical<T> BH = split<T>(B, A.dim[0], A.dim[1], true);
  tpmqrt(V, S, A, BH, trans);
  B = Dense<T>(BH);
}

define_method(
  void, tpmqrt_omm,
  (
    const Hierarchical<float>& V, const Hierarchical<float>& T, Hierarchical<float>& A, Dense<float>& B,
    bool trans
  )
) {
  tpmqrt_h_h_h_d(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (
    const Hierarchical<double>& V, const Hierarchical<double>& T, Hierarchical<double>& A, Dense<double>& B,
    bool trans
  )
) {
  tpmqrt_h_h_h_d(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_d_d_d_lr(const Dense<T>& V, const Dense<T>& S, Dense<T>& A, LowRank<T>& B, bool trans) {
  Dense<T> C(A);
  Dense<T> Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(S, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = S*C or S^t*C
  gemm(
    Dense<T>(identity, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense<float>& V, const Dense<float>& T, Dense<float>& A, LowRank<float>& B, bool trans)
) {
  tpmqrt_d_d_d_lr(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const Dense<double>& V, const Dense<double>& T, Dense<double>& A, LowRank<double>& B, bool trans)
) {
  tpmqrt_d_d_d_lr(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_lr_d_d_lr(const LowRank<T>& V, const Dense<T>& S, Dense<T>& A, LowRank<T>& B, bool trans) {
  Dense<T> C(A);
  LowRank<T> Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t * B
  trmm(S, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = S*C or S^t*C
  gemm(
    Dense<T>(identity, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const LowRank<float>& V, const Dense<float>& T, Dense<float>& A, LowRank<float>& B, bool trans)
) {
  tpmqrt_lr_d_d_lr(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const LowRank<double>& V, const Dense<double>& T, Dense<double>& A, LowRank<double>& B, bool trans)
) {
  tpmqrt_lr_d_d_lr(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_d_d_lr_lr(const Dense<T>& V, const Dense<T>& S, LowRank<T>& A, LowRank<T>& B, bool trans) {
  LowRank<T> C(A);
  Dense<T> Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(S, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = S*C or S^t*C
  gemm(
    Dense<T>(identity, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense<float>& V, const Dense<float>& T, LowRank<float>& A, LowRank<float>& B, bool trans)
) {
  tpmqrt_d_d_lr_lr(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const Dense<double>& V, const Dense<double>& T, LowRank<double>& A, LowRank<double>& B, bool trans)
) {
  tpmqrt_d_d_lr_lr(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_lr_d_lr_lr(const LowRank<T>& V, const Dense<T>& S, LowRank<T>& A, LowRank<T>& B, bool trans) {
  LowRank<T> C(A);
  LowRank<T> Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(S, C, 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = S*C or S^t*C
  gemm(
    Dense<T>(identity, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const LowRank<float>& V, const Dense<float>& T, LowRank<float>& A, LowRank<float>& B, bool trans)
) {
  tpmqrt_lr_d_lr_lr(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const LowRank<double>& V, const Dense<double>& T, LowRank<double>& A, LowRank<double>& B, bool trans)
) {
  tpmqrt_lr_d_lr_lr(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_d_d_d_h(const Dense<T>& V, const Dense<T>& S, Dense<T>& A, Hierarchical<T>& B, bool trans) {
  Dense<T> C(A);
  Dense<T> Vt = transpose(V);
  Dense<T> S_upper_tri(S);
  for(int64_t i=0; i<S_upper_tri.dim[0]; i++)
    for(int64_t j=0; j<i; j++)
      S_upper_tri(i, j) = 0.0;
  gemm(Vt, B, C, 1, 1); // C = A + Y^t*B
  if(trans) S_upper_tri = transpose(S_upper_tri);
  gemm(S_upper_tri, C, A, -1, 1); // A = A - (S or St)*C
  Dense<T> VSt(V.dim[0], S_upper_tri.dim[1]);
  gemm(V, S_upper_tri, VSt, 1, 1);
  gemm(VSt, C, B, -1, 1); // B = B - V*(S or St)*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense<float>& V, const Dense<float>& T, Dense<float>& A, Hierarchical<float>& B, bool trans)
) {
  tpmqrt_d_d_d_h(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const Dense<double>& V, const Dense<double>& T, Dense<double>& A, Hierarchical<double>& B, bool trans)
) {
  tpmqrt_d_d_d_h(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_h_h_d_h(const Hierarchical<T>& V, const Hierarchical<T>& S, Dense<T>& A, Hierarchical<T>& B, bool trans) {
  Hierarchical<T> HA = split<T>(A, B.dim[0], B.dim[1], true);
  tpmqrt(V, S, HA, B, trans);
  A = Dense<T>(HA);
}

define_method(
  void, tpmqrt_omm,
  (
    const Hierarchical<float>& V, const Hierarchical<float>& T, Dense<float>& A, Hierarchical<float>& B,
    bool trans
  )
) {
  tpmqrt_h_h_d_h(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (
    const Hierarchical<double>& V, const Hierarchical<double>& T, Dense<double>& A, Hierarchical<double>& B,
    bool trans
  )
) {
  tpmqrt_h_h_d_h(V, T, A, B, trans);
}

template<typename T>
void tpmqrt_h_h_h_h(const Hierarchical<T>& V, const Hierarchical<T>& S, Hierarchical<T>& A, Hierarchical<T>& B, bool trans) {
  if(trans) {
    for(int64_t i = 0; i < B.dim[0]; i++) {
      for(int64_t j = 0; j < B.dim[1]; j++) {
        for(int64_t k = 0; k < B.dim[1]; k++) {
          tpmqrt(V(i, j), S(i, j), A(j, k), B(i, k), trans);
        }
      }
    }
  }
  else {
    for(int64_t i = B.dim[0]-1; i >= 0; i--) {
      for(int64_t j = B.dim[1]-1; j >= 0; j--) {
        for(int64_t k = B.dim[1]-1; k >= 0; k--) {
          tpmqrt(V(i, j), S(i, j), A(j, k), B(i, k), trans);
        }
      }
    }
  }
}

define_method(
  void, tpmqrt_omm,
  (
    const Hierarchical<float>& V, const Hierarchical<float>& T,
    Hierarchical<float>& A, Hierarchical<float>& B,
    bool trans
  )
) {
  tpmqrt_h_h_h_h(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (
    const Hierarchical<double>& V, const Hierarchical<double>& T,
    Hierarchical<double>& A, Hierarchical<double>& B,
    bool trans
  )
) {
  tpmqrt_h_h_h_h(V, T, A, B, trans);
}

// Fallback default, abort with error message
define_method(
  void, tpmqrt_omm,
  (const Matrix& V, const Matrix& T, Matrix& A, Matrix& B, bool)
) {
  omm_error_handler("tpmqrt", {V, T, A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma

