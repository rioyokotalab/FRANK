#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/node.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

std::tuple<Dense, std::vector<int64_t>> geqp3(Node& A) { return geqp3_omm(A); }

// Fallback default, abort with error message
define_method(DenseIndexSetPair, geqp3_omm, (Dense& A)) {
  timing::start("DGEQP3");
  // TODO The 0 initial value is important! Otherwise axes are fixed and results
  // can be wrong. See netlib dgeqp3 reference.
  // However, much faster with -1... maybe better starting values exist?
  std::vector<int> jpvt(A.dim[1], 0);
  std::vector<double> tau(std::min(A.dim[0], A.dim[1]), 0);
  LAPACKE_dgeqp3(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &jpvt[0], &tau[0]
  );
  // jpvt is 1-based, bad for indexing!
  std::vector<int64_t> column_order(jpvt.size());
  for (size_t i=0; i<jpvt.size(); ++i) column_order[i] = jpvt[i] - 1;
  timing::start("R construction");
  Dense R(A.dim[1], A.dim[1]);
  for(int64_t i=0; i<std::min(A.dim[0], R.dim[0]); i++) {
    for(int64_t j=i; j<R.dim[1]; j++) {
      R(i, j) = A(i, j);
    }
  }timing::stop("R construction");
  timing::stop("DGEQP3");
  return {std::move(R), std::move(column_order)};
}

// Fallback default, abort with error message
define_method(DenseIndexSetPair, geqp3_omm, (Node& A)) {
  omm_error_handler("geqp3", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
