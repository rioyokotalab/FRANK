#include "hicma/operations/geqp3.h"

#include "hicma/node.h"
#include "hicma/dense.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

std::vector<int> geqp3(Node& A, Node& Q, Node& R) {
  return geqp3_omm(A, Q, R);
}

// Fallback default, abort with error message
BEGIN_SPECIALIZATION(
  geqp3_omm, std::vector<int>,
  Dense& A, Dense& Q, Dense& R
) {
  assert(A.dim[0] == Q.dim[0]);
  assert(Q.dim[1] == R.dim[0]);
  assert(A.dim[1] == R.dim[1]);
  // Pivoted QR
  for (int i=0; i<std::min(A.dim[0], A.dim[1]); i++) Q(i, i) = 1.0;
  std::vector<int> jpvt(A.dim[1], 0);
  std::vector<double> tau(std::min(A.dim[0], A.dim[1]));
  LAPACKE_dgeqp3(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A[0], A.dim[1],
    &jpvt[0], &tau[0]
  );
  LAPACKE_dormqr(
    LAPACK_ROW_MAJOR,
    'L', 'N',
    A.dim[0], A.dim[1], A.dim[0],
    &A[0], A.dim[0],
    &tau[0],
    &Q[0], Q.dim[1]
  );
  // jpvt is 1-based, bad for indexing!
  for (int& i : jpvt) --i;
  for(int i=0; i<std::min(A.dim[0], A.dim[1]); i++) {
    for(int j=0; j<A.dim[1]; j++) {
      if (j >= i) R(i, j) = A(i, j);
    }
  }
  return jpvt;
} END_SPECIALIZATION;

// Fallback default, abort with error message
BEGIN_SPECIALIZATION(
  geqp3_omm, std::vector<int>,
  Node& A, Node& Q, Node& R
) {
  std::cerr << "geqp3(";
  std::cerr << A.type() << "," << Q.type() << "," << R.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
