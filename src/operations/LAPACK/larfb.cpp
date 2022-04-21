#include "FRANK/operations/LAPACK.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
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


namespace FRANK
{

declare_method(
  void, larfb_omm,
  (virtual_<const Matrix&>, virtual_<const Matrix&>, virtual_<Matrix&>, const bool)
)

void larfb(const Matrix& V, const Matrix& T, Matrix& C, const bool trans) {
  larfb_omm(V, T, C, trans);
}

define_method(
  void, larfb_omm,
  (const Dense& V, const Dense& T, Dense& C, const bool trans)
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
  (const Dense& V, const Dense& T, LowRank& C, const bool trans)
) {
  larfb(V, T, C.U, trans);
}

// Fallback default, abort with error message
define_method(
  void, larfb_omm, (const Matrix& V, const Matrix& T, Matrix& C, const bool)
) {
  omm_error_handler("larfb", {V, T, C}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
