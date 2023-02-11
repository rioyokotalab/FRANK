#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/empty.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>
#include <omp.h>


namespace hicma
{

std::tuple<MatrixProxy, MatrixProxy> getrf(Matrix& A) {
  std::tuple<MatrixProxy, MatrixProxy> out = getrf_omm(A);
  return out;
}

template<typename T>
MatrixPair hierarchical_getrf(Hierarchical<T>& A) {
  Hierarchical<T> L(A.dim[0], A.dim[1]);
  Hierarchical<T> A_copy(A);
  Hierarchical<T> L_copy(A.dim[0], A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; i++) {
    std::cout<<"Iteration "<<i<<std::endl;
    #pragma omp task depend(inout: A.lock[i][i])
      {std::cout<<"race ";}
    #pragma omp task depend(inout: A.lock[i][i])
      {std::cout<<"car ";} 
    #pragma omp task depend(inout: A.lock[i][i])
    {
      MatrixPair test = getrf_omm(A(i, i));
      //std::tie(A_copy(i, i), A_copy(i, i)) = test;
    }
    //#pragma omp task shared(A, L) depend(inout: A.lock[i][i])
    {
      std::tie(L(i, i), A(i, i)) = getrf_omm(A(i, i));
    }
    for (int64_t i_c=i+1; i_c<L.dim[0]; i_c++) {
     /* #pragma omp task shared(L_copy) depend(in: A.lock[0][0]) depend(inout: A.lock[1][0])
      {
        L_copy(i_c, i) = std::move(A_copy(i_c, i));
        A_copy(i_c, i) = Empty(get_n_rows(L_copy(i_c, i)), get_n_cols(L_copy(i_c, i)));
        trsm(A_copy(i, i), L_copy(i_c, i), TRSM_UPPER, TRSM_RIGHT);
      }*/
      L(i_c, i) = std::move(A(i_c, i));
      A(i_c, i) = Empty(get_n_rows(L(i_c, i)), get_n_cols(L(i_c, i)));
      //#pragma omp task depend(in: A.lock[0][0]) depend(inout: A.lock[1][0])
      {
        trsm(A(i, i), L(i_c, i), TRSM_UPPER, TRSM_RIGHT);
      }
    }
    for (int64_t j=i+1; j<A.dim[1]; j++) {
      /*#pragma omp task shared(L_copy) depend(in: A.lock[0][0]) depend(inout: A.lock[0][1])
      {
        L_copy(i, j) = Empty(get_n_rows(A(i, j)), get_n_cols(A_copy(i, j)));
        trsm(L_copy(i, i), A_copy(i, j), TRSM_LOWER, TRSM_LEFT);
      }*/
      L(i, j) = Empty(get_n_rows(A(i, j)), get_n_cols(A(i, j)));
      //#pragma omp task depend(in: A.lock[0][0]) depend(inout: A.lock[0][1])
      {
        trsm(L(i, i), A(i, j), TRSM_LOWER, TRSM_LEFT);
      }
    }
    for (int64_t i_c=i+1; i_c<L.dim[0]; i_c++) {
      for (int64_t k=i+1; k<A.dim[1]; k++) {
       /* #pragma omp task shared(L_copy) depend(in: A.lock[0][1], A.lock[1][0]) depend(inout: A.lock[1][1])
        {
          gemm(L_copy(i_c, i), A_copy(i, k), A_copy(i_c, k), -1, 1);
        }*/
        //#pragma omp task depend(in: A.lock[0][1], A.lock[1][0]) depend(inout: A.lock[1][1])
        {
          gemm(L(i_c, i), A(i, k), A(i_c, k), -1, 1);
        }
      }
    }
  }
  //#pragma omp taskwait
  return {std::move(L), std::move(A)};
  //return {std::move(L_copy), std::move(A_copy)};
}

define_method(MatrixPair, getrf_omm, (Hierarchical<float>& A)) {
  return hierarchical_getrf(A);
}

define_method(MatrixPair, getrf_omm, (Hierarchical<double>& A)) {
  return hierarchical_getrf(A);
}

// single precision
define_method(MatrixPair, getrf_omm, (Dense<float>& A)) {
  timing::start("SGETRF");
  Dense<float> L(A.dim[0], A.dim[1]);
  //std::vector<int> ipiv(std::min(A.dim[0], A.dim[1]));
  LAPACKE_mkl_sgetrfnpi(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1], std::min(A.dim[0], A.dim[1]),
    &A, A.stride
  );
  /*LAPACKE_sgetrf(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &ipiv[0]
  );*/
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<i; j++) {
      L(i, j) = A(i, j);
      A(i, j) = 0;
    }
    L(i, i) = 1;
  }
  timing::stop("SGETRF");
  return {std::move(L), std::move(A)};
}

// double precision
define_method(MatrixPair, getrf_omm, (Dense<double>& A)) {
  timing::start("DGETRF");
  Dense<double> L(A.dim[0], A.dim[1]);
  //std::vector<int> ipiv(std::min(A.dim[0], A.dim[1]));
  LAPACKE_mkl_dgetrfnpi(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1], std::min(A.dim[0], A.dim[1]),
    &A, A.stride
  );
  /*LAPACKE_dgetrf(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &ipiv[0]
  );*/
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<i; j++) {
      L(i, j) = A(i, j);
      A(i, j) = 0;
    }
    L(i, i) = 1;
  }
  timing::stop("DGETRF");
  return {std::move(L), std::move(A)};
}

define_method(MatrixPair, getrf_omm, (Matrix& A)) {
  omm_error_handler("getrf", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
