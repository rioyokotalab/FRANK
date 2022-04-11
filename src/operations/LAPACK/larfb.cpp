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
#include <cblas.h>
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
  (const Dense& V, const Dense& T, LowRank& C, bool trans)
) {
  larfb(V, T, C.U, trans);
}

// Fallback default, abort with error message
define_method(
  void, larfb_omm, (const Matrix& V, const Matrix& T, Matrix& C, bool)
) {
  omm_error_handler("larfb", {V, T, C}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
