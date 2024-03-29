#include "FRANK/operations/LAPACK.h"

#include "FRANK/classes/dense.h"
#include "FRANK/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>


namespace FRANK
{

std::tuple<Dense, Dense, Dense> svd(Dense& A) {
  const int64_t dim_min = std::min(A.dim[0], A.dim[1]);
  Dense U(A.dim[0], dim_min);
  Dense S(dim_min, dim_min);
  Dense V(dim_min, A.dim[1]);
  std::vector<double> Sdiag(S.dim[0], 0);
  std::vector<double> work(S.dim[0]-1, 0);
  LAPACKE_dgesvd(
    LAPACK_ROW_MAJOR,
    'S', 'S',
    A.dim[0], A.dim[1],
    &A, A.stride,
    &Sdiag[0],
    &U, U.stride,
    &V, V.stride,
    &work[0]
  );
  for(int64_t i=0; i<S.dim[0]; i++){
    S(i, i) = Sdiag[i];
  }
  return {std::move(U), std::move(S), std::move(V)};
}

std::vector<double> get_singular_values(Dense& A) {
  std::vector<double> Sdiag(std::min(A.dim[0], A.dim[1]), 1);
  Dense work(A.dim[1]-1,1);
  // Since we use 'N' we can avoid allocating memory for U and V
  LAPACKE_dgesvd(
    LAPACK_ROW_MAJOR,
    'N', 'N',
    A.dim[0], A.dim[1],
    &A, A.stride,
    Sdiag.data(),
    &work, A.stride,
    &work, A.stride,
    &work
  );
  return Sdiag;
}

} // namespace FRANK
