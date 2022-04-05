#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

void qr(Matrix& A, Matrix& Q, Matrix& R) {
  // TODO consider moving assertions here (same in other files)!
  qr_omm(A, Q, R);
}

Dense get_right_factor(const Matrix& A) {
  return get_right_factor_omm(A);
}

void update_right_factor(Matrix& A, Matrix& R) {
  update_right_factor_omm(A, R);
}


define_method(void, qr_omm, (Dense& A, Dense& Q, Dense& R)) {
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == R.dim[0]);
  assert(R.dim[1] == A.dim[1]);
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<double> tau(k);
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
  // Copy upper triangular (or trapezoidal) part of A into R
  for(int64_t i=0; i<std::min(A.dim[0], R.dim[0]); i++) {
    for(int64_t j=i; j<R.dim[1]; j++) {
      R(i, j) = A(i, j);
    }
  }
  // Copy strictly lower triangular (or trapezoidal) part of A into Q
  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=0; j<std::min(i, A.dim[1]); j++) {
      Q(i, j) = A(i, j);
    }
  }
  // TODO Consider making special function for this. Performance heavy
  // and not always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense derivative that remains in elementary
  // reflector form, uses dormqr instead of gemm and can be transformed to
  // Dense via dorgqr!
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]);
}

define_method(void, qr_omm, (Matrix& A, Matrix& Q, Matrix& R)) {
  omm_error_handler("qr", {A, Q, R}, __FILE__, __LINE__);
  std::abort();
}

define_method(Dense, get_right_factor_omm, (const Dense& A)) {
  return Dense(A);
}

define_method(Dense, get_right_factor_omm, (const LowRank& A)) {
  Dense SV = gemm(A.S, A.V);
  return SV;
}

define_method(
  void, update_right_factor_omm,
  (Dense& A, Dense& R)
) {
  A = std::move(R);
}

define_method(
  void, update_right_factor_omm,
  (LowRank& A, Dense& R)
) {
  A.S = 0.0;
  for(int64_t i=0; i<std::min(A.S.dim[0], A.S.dim[1]); i++) {
    A.S(i, i) = 1.0;
  }
  A.V = std::move(R);
}

void triangularize_block_col(int64_t j, Hierarchical& A, Hierarchical& T) {
  //Put right factors of Aj into Rj
  Hierarchical Rj(A.dim[0]-j, 1);
  for(int64_t i=0; i<Rj.dim[0]; i++) {
    Rj(i, 0) = get_right_factor(A(j+i, j));
  }
  //QR on concatenated right factors
  Dense DRj(Rj);
  Dense Tj(DRj.dim[1], DRj.dim[1]);
  geqrt(DRj, Tj);
  T(j, 0) = std::move(Tj);
  //Slice DRj based on Rj
  int64_t rowOffset = 0;
  for(int64_t i=0; i<Rj.dim[0]; i++) {
    assert(DRj.dim[1] == get_n_cols(Rj(i, 0)));
    int64_t dim_Rij[2]{get_n_rows(Rj(i, 0)), get_n_cols(Rj(i, 0))};
    Dense Rij(dim_Rij[0], dim_Rij[1]);
    DRj.copy_to(Rij, rowOffset);
    Rj(i, 0) = std::move(Rij);
    rowOffset += dim_Rij[0];
  }
  //Multiply block householder vectors with respective left factors
  for(int64_t i=0; i<Rj.dim[0]; i++) {
    update_right_factor(A(j+i, j), Rj(i, 0));
  }
}

void apply_block_col_householder(const Hierarchical& Y, const Hierarchical& T, int64_t k, bool trans, Hierarchical& A, int64_t j) {
  assert(A.dim[0] == Y.dim[0]);
  Hierarchical YkT(1, Y.dim[0]-k);
  for(int64_t i=0; i<YkT.dim[1]; i++)
    YkT(0, i) = transpose(Y(i+k,k));

  Hierarchical C(1, 1);
  C(0, 0) = A(k, j); //C = Akj
  trmm(Y(k, k), C(0, 0), 'l', 'l', 't', 'u', 1); //C = Ykk^T x Akj
  for(int64_t i=k+1; i<A.dim[0]; i++) {
    gemm(YkT(0, i-k), A(i, j), C(0, 0), 1, 1); //C += Yik^T x Aij
  }
  trmm(T(k, 0), C(0, 0), 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = (T or T^T) x C
  for(int64_t i=k; i<A.dim[0]; i++) {
    //Aij = Aij - Yik x C
    if(i == k) { //Use trmm since Ykk is unit lower triangular
      Hierarchical _C(C);
      trmm(Y(k, k), _C(0, 0), 'l', 'l', 'n', 'u', 1);
      gemm(
        Dense(identity, {}, get_n_rows(_C(0, 0)), get_n_rows(_C(0, 0))),
        _C(0, 0), A(k, j), -1, 1
      );
    }
    else { //Use gemm otherwise
      gemm(Y(i, k), C(0, 0), A(i, j), -1, 1);
    }
  }
}

void blocked_householder_blr_qr(Hierarchical& A, Hierarchical& T) {
  assert(T.dim[0] == A.dim[1]);
  assert(T.dim[1] == 1);
  for(int64_t k = 0; k < A.dim[1]; k++) {
    triangularize_block_col(k, A, T);
    for(int64_t j = k+1; j < A.dim[1]; j++) {
      apply_block_col_householder(A, T, k, true, A, j);
    }
  }
}

void left_multiply_blocked_reflector(const Hierarchical& Y, const Hierarchical& T, Hierarchical& C, bool trans) {
  if(trans) {
    for(int64_t k = 0; k < Y.dim[1]; k++) {
      for(int64_t j = k; j < Y.dim[1]; j++) {
        apply_block_col_householder(Y, T, k, trans, C, j);
      }
    }
  }
  else {
    for(int64_t k = Y.dim[1]-1; k >= 0; k--) {
      for(int64_t j = k; j < Y.dim[1]; j++) {
        apply_block_col_householder(Y, T, k, trans, C, j);
      }
    }
  }
}

void tiled_householder_blr_qr(Hierarchical& A, Hierarchical& T) {
  assert(T.dim[0] == A.dim[0]);
  assert(T.dim[1] == A.dim[1]);
  for(int64_t k = 0; k < A.dim[1]; k++) {
    geqrt(A(k, k), T(k, k));
    for(int64_t j = k+1; j < A.dim[1]; j++) {
      larfb(A(k, k), T(k, k), A(k, j), true);
    }
    for(int64_t i = k+1; i < A.dim[0]; i++) {
      tpqrt(A(k, k), A(i, k), T(i, k));
      for(int64_t j = k+1; j < A.dim[1]; j++) {
        tpmqrt(A(i, k), T(i, k), A(k, j), A(i, j), true);
      }
    }
  }
}

void left_multiply_tiled_reflector(const Hierarchical& Y, const Hierarchical& T, Hierarchical& C, bool trans) {
  if(trans) {
    for(int64_t k = 0; k < Y.dim[1]; k++) {
      for(int64_t j = k; j < Y.dim[1]; j++) {
        larfb(Y(k, k), T(k, k), C(k, j), trans);
      }
      for(int64_t i = k+1; i < Y.dim[0]; i++) {
        for(int64_t j = k; j < Y.dim[1]; j++) {
          tpmqrt(Y(i, k), T(i, k), C(k, j), C(i, j), trans);
        }
      }
    }
  }
  else {
    for(int64_t k = Y.dim[1]-1; k >= 0; k--) {
      for(int64_t i = Y.dim[0]-1; i > k; i--) {
        for(int64_t j = k; j < Y.dim[1]; j++) {
          tpmqrt(Y(i, k), T(i, k), C(k, j), C(i, j), trans);
        }
      }
      for(int64_t j = k; j < Y.dim[1]; j++) {
        larfb(Y(k, k), T(k, k), C(k, j), trans);
      }
    }
  }
}

} // namespace hicma
