#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"


namespace hicma
{

std::tuple<Dense, std::vector<int>> geqp3(Node& A) {
  return geqp3_omm(A);
}

// Fallback default, abort with error message
define_method(DenseIntVectorPair, geqp3_omm, (Dense& A)) {
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
  for (int& i : jpvt) --i;
  timing::start("R construction");
  Dense R(A.dim[1], A.dim[1]);
  for(int i=0; i<std::min(A.dim[0], R.dim[0]); i++) {
    for(int j=i; j<R.dim[1]; j++) {
      R(i, j) = A(i, j);
    }
  }timing::stop("R construction");
  timing::stop("DGEQP3");
  return {std::move(R), std::move(jpvt)};
}

// Fallback default, abort with error message
define_method(DenseIntVectorPair, geqp3_omm, (Node& A)) {
  std::cerr << "geqp3(" << A.type() << ") undefined." << std::endl;
  abort();
}

} // namespace hicma
