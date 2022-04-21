#include "FRANK/operations/LAPACK.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/functions.h"
#include "FRANK/operations/BLAS.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <cstdlib>
#include <vector>


namespace FRANK
{

declare_method(
  void, tpmqrt_omm,
  (
    virtual_<const Matrix&>, virtual_<const Matrix&>,
    virtual_<Matrix&>, virtual_<Matrix&>,
    const bool
  )
)

void tpmqrt(
  const Matrix& V, const Matrix& T, Matrix& A, Matrix& B, const bool trans
) {
  tpmqrt_omm(V, T, A, B, trans);
}

define_method(
  void, tpmqrt_omm,
  (const Dense& V, const Dense& T, Dense& A, Dense& B, const bool trans)
) {
  // D D D D
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
  (const LowRank& V, const Dense& T, Dense& A, Dense& B, const bool trans)
) {
  // LR D D D
  Dense C(A);
  LowRank Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, Side::Left, Mode::Upper, trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, {}, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense& V, const Dense& T, LowRank& A, Dense& B, const bool trans)
) {
  // D D LR D
  Dense C(A);
  gemm(V, B, C, 1, 1, true, false); //C = A + Y^t*B
  trmm(T, C, Side::Left, Mode::Upper, trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, {}, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C //Recompression
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense& V, const Dense& T, Dense& A, LowRank& B, const bool trans)
) {
  // D D D LR
  Dense C(A);
  Dense Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, Side::Left, Mode::Upper, trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, {}, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const LowRank& V, const Dense& T, LowRank& A, Dense& B, const bool trans)
) {
  // LR D LR D
  LowRank C(A);
  LowRank Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, Side::Left, Mode::Upper, trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, {}, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const LowRank& V, const Dense& T, Dense& A, LowRank& B, const bool trans)
) {
  // LR D D LR
  Dense C(A);
  LowRank Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t * B
  trmm(T, C, Side::Left, Mode::Upper, trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, {}, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const Dense& V, const Dense& T, LowRank& A, LowRank& B, const bool trans)
) {
  // D D LR LR
  LowRank C(A);
  Dense Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, Side::Left, Mode::Upper, trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, {}, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

define_method(
  void, tpmqrt_omm,
  (const LowRank& V, const Dense& T, LowRank& A, LowRank& B, const bool trans)
) {
  // LR D LR LR
  LowRank C(A);
  LowRank Vt = transpose(V);
  gemm(Vt, B, C, 1, 1); //C = A + Y^t*B
  trmm(T, C, Side::Left, Mode::Upper, trans ? 't' : 'n', 'n', 1); //C = T*C or T^t*C
  gemm(
    Dense(identity, {}, C.dim[0], C.dim[0]),
    C, A, -1, 1
  ); //A = A - I*C
  gemm(V, C, B, -1, 1); //B = B - Y*C
}

// Fallback default, abort with error message
define_method(
  void, tpmqrt_omm,
  (const Matrix& V, const Matrix& T, Matrix& A, Matrix& B, const bool)
) {
  omm_error_handler("tpmqrt", {V, T, A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK

