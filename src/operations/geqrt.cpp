#include "hicma/operations/geqrt.h"

#include "hicma/node.h"
#include "hicma/node_proxy.h"
#include "hicma/dense.h"
#include "hicma/hierarchical.h"
#include "hicma/operations/tpmqrt.h"
#include "hicma/operations/tpqrt.h"
#include "hicma/operations/larfb.h"

#include <cassert>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

void geqrt(NodeProxy& A, NodeProxy& T) {
  geqrt(*A.ptr, *T.ptr);
}
void geqrt(Node& A, NodeProxy& T) {
  geqrt(A, *T.ptr);
}
void geqrt(NodeProxy& A, Node& T) {
  geqrt(*A.ptr, T);
}

void geqrt(Node& A, Node& T) {
  geqrt_omm(A, T);
}

BEGIN_SPECIALIZATION(geqrt_omm, void, Dense& A, Dense& T) {
  assert(T.dim[0] == A.dim[1]);
  assert(T.dim[1] == A.dim[1]);
  LAPACKE_dgeqrt3(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A[0], A.dim[1],
    &T[0], T.dim[1]
  );
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(geqrt_omm, void, Hierarchical& A, Hierarchical&T) {
  std::cerr << "Possibly not fully implemented yet. Read code!!!" << std::endl;
  for(int k = 0; k < A.dim[1]; k++) {
    geqrt(A(k, k), T(k, k));
    for(int j = k+1; j < A.dim[1]; j++) {
      larfb(A(k, k), T(k, k), A(k, j), true);
    }
    int dim0 = -1;
    int dim1 = -1;
    for(int i = k+1; i < A.dim[0]; i++) {
      tpqrt(A(k, k), A(i, k), T(i, k));
      for(int j = k+1; j < A.dim[1]; j++) {
        tpmqrt(A(i, k), T(i, k), A(k, j), A(i, j), true);
      }
    }
  }
} END_SPECIALIZATION;

void geqrt2(Dense& A, Dense& T) {
  assert(T.dim[0] == A.dim[1]);
  assert(T.dim[1] == A.dim[1]);
  LAPACKE_dgeqrt2(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A[0], A.dim[1],
    &T[0], T.dim[1]
  );
}

} // namespace hicma
