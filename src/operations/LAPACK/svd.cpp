#include "hicma/operations/LAPACK/svd.h"

#include "hicma/classes/dense.h"
#include "hicma/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <algorithm>
#include <tuple>
#include <utility>

namespace hicma
{

std::tuple<Dense, Dense, Dense> svd(Dense& A) {
  start("-DGESVD");
  int dim_min = std::min(A.dim[0], A.dim[1]);
  Dense Sdiag(dim_min, 1);
  Dense work(dim_min-1, 1);
  Dense U(A.dim[0], dim_min);
  Dense V(dim_min, A.dim[1]);
  LAPACKE_dgesvd(
    LAPACK_ROW_MAJOR,
    'S', 'S',
    A.dim[0], A.dim[1],
    &A[0], A.dim[1],
    &Sdiag[0],
    &U[0], U.dim[1],
    &V[0], V.dim[1],
    &work[0]
  );
  Dense S(dim_min, dim_min);
  for(int i=0; i<dim_min; i++){
    S(i, i) = Sdiag[i];
  }
  stop("-DGESVD",false);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense, Dense> sdd(Dense& A) {
  start("-DGESDD");
  int dim_min = std::min(A.dim[0], A.dim[1]);
  Dense Sdiag(dim_min, 1);
  Dense work(dim_min-1, 1);
  Dense U(A.dim[0], dim_min);
  Dense V(dim_min, A.dim[1]);
  // dgesdd is faster, but makes little/no difference in randomized SVD
  LAPACKE_dgesdd(
    LAPACK_ROW_MAJOR,
    'S',
    A.dim[0], A.dim[1],
    &A[0], A.dim[1],
    &Sdiag[0],
    &U[0], U.dim[1],
    &V[0], V.dim[1]
  );
  Dense S(dim_min, dim_min);
  for(int i=0; i<dim_min; i++){
    S(i, i) = Sdiag[i];
  }
  stop("-DGESDD", false);
  return {std::move(U), std::move(S), std::move(V)};
}

Dense get_singular_values(Dense& A) {
  Dense Sdiag(std::min(A.dim[0], A.dim[1]), 1);
  Dense work(A.dim[1]-1,1);
  // Since we use 'N' we can avoid allocating memory for U and V
  LAPACKE_dgesvd(
    LAPACK_ROW_MAJOR,
    'N', 'N',
    A.dim[0], A.dim[1],
    &A[0], A.dim[1],
    &Sdiag[0],
    &work[0], A.dim[0],
    &work[0], A.dim[1],
    &work[0]
  );
  return Sdiag;
}

} // namespace hicma
