#include "FRANK/operations/LAPACK.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/util/omm_error_handler.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>


namespace FRANK
{

declare_method(
  void, geqrt_omm,
  (virtual_<Matrix&>, virtual_<Matrix&>)
)

void geqrt(Matrix& A, Matrix& T) { geqrt_omm(A, T); }

define_method(void, geqrt_omm, (Dense& A, Dense& T)) {
  assert(T.dim[0] == A.dim[1]);
  assert(T.dim[1] == A.dim[1]);
  LAPACKE_dgeqrt3(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &T, T.stride
  );
}

// Fallback default, abort with error message
define_method(void, geqrt_omm, (Matrix& A, Matrix& T)) {
  omm_error_handler("geqrt", {A, T}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
