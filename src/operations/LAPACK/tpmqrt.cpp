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
#include <cblas.h>
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
  (const LowRank& V, const Dense& T, Dense& A, Dense& B, bool trans)
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
  (const Dense& V, const Dense& T, LowRank& A, Dense& B, bool trans)
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
  (const Dense& V, const Dense& T, Dense& A, LowRank& B, bool trans)
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
  (const LowRank& V, const Dense& T, LowRank& A, Dense& B, bool trans)
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
  (const LowRank& V, const Dense& T, Dense& A, LowRank& B, bool trans)
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
  (const Dense& V, const Dense& T, LowRank& A, LowRank& B, bool trans)
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
  (const LowRank& V, const Dense& T, LowRank& A, LowRank& B, bool trans)
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
  (const Matrix& V, const Matrix& T, Matrix& A, Matrix& B, bool)
) {
  omm_error_handler("tpmqrt", {V, T, A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma

