#include "hicma/operations/LAPACK.h"

#include "hicma/classes/dense.h"
#include "hicma/util/timer.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>
#include <iostream>


namespace hicma
{

template<>
std::tuple<Dense<double>, Dense<double>, Dense<double>> svd(Dense<double>& A) {
  timing::start("DGESVD");
  int64_t dim_min = std::min(A.dim[0], A.dim[1]);
  Dense<double> U(A.dim[0], dim_min);
  Dense<double> S(dim_min, dim_min);
  Dense<double> V(dim_min, A.dim[1]);
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
  timing::stop("DGESVD");
  return {std::move(U), std::move(S), std::move(V)};
}

template<>
std::tuple<Dense<float>, Dense<float>, Dense<float>> svd(Dense<float>& A) {
  timing::start("SGESVD");
  int64_t dim_min = std::min(A.dim[0], A.dim[1]);
  Dense<float> U(A.dim[0], dim_min);
  Dense<float> S(dim_min, dim_min);
  Dense<float> V(dim_min, A.dim[1]);
  std::vector<float> Sdiag(S.dim[0], 0);
  std::vector<float> work(S.dim[0]-1, 0);
  LAPACKE_sgesvd(
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
  timing::stop("SGESVD");
  return {std::move(U), std::move(S), std::move(V)};
}

template<>
std::tuple<Dense<float>, Dense<float>, Dense<float>> sdd(Dense<float>& A) {
  timing::start("SGESDD");
  int64_t dim_min = std::min(A.dim[0], A.dim[1]);
  Dense<float> Sdiag(dim_min, 1);
  Dense<float> work(dim_min-1, 1);
  Dense<float> U(A.dim[0], dim_min);
  Dense<float> V(dim_min, A.dim[1]);
  // dgesdd is faster, but makes little/no difference in randomized SVD
  LAPACKE_sgesdd(
    LAPACK_ROW_MAJOR,
    'S',
    A.dim[0], A.dim[1],
    &A, A.stride,
    &Sdiag,
    &U, U.stride,
    &V, V.stride
  );
  Dense<float> S(dim_min, dim_min);
  for(int64_t i=0; i<dim_min; i++){
    S(i, i) = Sdiag[i];
  }
  timing::stop("SGESDD");
  return {std::move(U), std::move(S), std::move(V)};
}

template<>
std::tuple<Dense<double>, Dense<double>, Dense<double>> sdd(Dense<double>& A) {
  timing::start("DGESDD");
  int64_t dim_min = std::min(A.dim[0], A.dim[1]);
  Dense<double> Sdiag(dim_min, 1);
  Dense<double> work(dim_min-1, 1);
  Dense<double> U(A.dim[0], dim_min);
  Dense<double> V(dim_min, A.dim[1]);
  // dgesdd is faster, but makes little/no difference in randomized SVD
  LAPACKE_dgesdd(
    LAPACK_ROW_MAJOR,
    'S',
    A.dim[0], A.dim[1],
    &A, A.stride,
    &Sdiag,
    &U, U.stride,
    &V, V.stride
  );
  Dense<double> S(dim_min, dim_min);
  for(int64_t i=0; i<dim_min; i++){
    S(i, i) = Sdiag[i];
  }
  timing::stop("DGESDD");
  return {std::move(U), std::move(S), std::move(V)};
}

template<>
std::vector<float> get_singular_values(Dense<float>& A) {
  std::vector<float> Sdiag(std::min(A.dim[0], A.dim[1]), 1);
  Dense<float> work(A.dim[1]-1,1);
  // Since we use 'N' we can avoid allocating memory for U and V
  LAPACKE_sgesvd(
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

template<>
std::vector<double> get_singular_values(Dense<double>& A) {
  std::vector<double> Sdiag(std::min(A.dim[0], A.dim[1]), 1);
  Dense<double> work(A.dim[1]-1,1);
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

template<>
void inverse(Dense<double>& A) {
  std::vector<int> ipiv(std::min(A.dim[0], A.dim[1]));
  int info;
  info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, ipiv.data());
  if (info != 0) {
    std::cout << "DGETRF failed in inverse().\n";
  }
  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, A.dim[0], &A, A.stride, ipiv.data());
  if (info != 0) {
    std::cout << "DGETRI failed in inverse().\n";
  }
}

template<>
void inverse(Dense<float>& A) {
  std::vector<int> ipiv(std::min(A.dim[0], A.dim[1]));
  int info;
  info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, ipiv.data());
  if (info != 0) {
    std::cout << "DGETRF failed in inverse().\n";
  }
  info = LAPACKE_sgetri(LAPACK_ROW_MAJOR, A.dim[0], &A, A.stride, ipiv.data());
  if (info != 0) {
    std::cout << "DGETRI failed in inverse().\n";
  }
}

} // namespace hicma
