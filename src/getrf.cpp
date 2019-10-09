#include "hicma/operations.h"

#include "hicma/dense.h"
#include "hicma/hierarchical.h"
#include "hicma/util/timer.h"

#include "yorel/multi_methods.hpp"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

namespace hicma
{

void getrf(Any& A) {
  getrf_omm(*A.ptr.get());
}

void getrf(Node& A) {
  getrf_omm(A);
}

BEGIN_SPECIALIZATION(
  getrf_omm, void,
  Hierarchical& A
) {
  for (int i=0; i<A.dim[0]; i++) {
    getrf(A(i,i));
    for (int j=i+1; j<A.dim[0]; j++) {
      A(i,j).trsm(A(i,i), 'l');
      A(j,i).trsm(A(i,i), 'u');
    }
    for (int j=i+1; j<A.dim[0]; j++) {
      for (int k=i+1; k<A.dim[0]; k++) {
        A(j,k).gemm(A(j,i), A(i,k), -1, 1);
      }
    }
  }
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(getrf_omm, void, Dense& A) {
  start("-DGETRF");
  std::vector<int> ipiv(std::min(A.dim[0], A.dim[1]));
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A.data[0], A.dim[1], &ipiv[0]);
  stop("-DGETRF",false);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(getrf_omm, void, Node& A) {
  std::cout << "getrf(" << A.type() << ") not defined!" << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
