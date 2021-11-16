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
#include "hicma/util/pre_scheduler.h"
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

std::tuple<Dense, Dense> make_left_orthogonal(const Matrix& A) {
  return make_left_orthogonal_omm(A);
}

void update_splitted_size(const Matrix& A, int64_t& rows, int64_t& cols) {
  update_splitted_size_omm(A, rows, cols);
}

MatrixProxy split_by_column(
  const Matrix& A, Matrix& storage, int64_t &currentRow
) {
  return split_by_column_omm(A, storage, currentRow);
}

MatrixProxy concat_columns(
  const Matrix& A, const Matrix& splitted, const Matrix& Q, int64_t& currentRow
) {
  return concat_columns_omm(A, splitted, Q, currentRow);
}

void orthogonalize_block_col(int64_t j, const Matrix& A, Matrix& Q, Matrix& R) {
  orthogonalize_block_col_omm(j, A, Q, R);
}

Dense get_right_factor(const Matrix& A) {
  return get_right_factor_omm(A);
}

void update_right_factor(Matrix& A, Matrix& R) {
  update_right_factor_omm(A, R);
}


define_method(void, qr_omm, (Dense& A, Dense& Q, Dense& R)) {
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == A.dim[1]);
  assert(R.dim[0] == A.dim[1]);
  assert(R.dim[1] == A.dim[1]);
  timing::start("QR");
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<double> tau(k);
  for (int64_t i=0; i<std::min(Q.dim[0], Q.dim[1]); i++) Q(i, i) = 1.0;
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
  // TODO Consider using A for the dorgqr and moving to Q afterwards! That
  // also simplify this loop.
  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=i; j<Q.dim[1]; j++) {
      R(i, j) = A(i, j);
    }
  }
  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=0; j<std::min(i, Q.dim[1]); j++) {
      Q(i, j) = A(i, j);
    }
  }
  // TODO Consider making special function for this. Performance heavy
  // and not always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense derivative that remains in elementary
  // reflector form, uses dormqr instead of gemm and can be transformed to
  // Dense via dorgqr!
  LAPACKE_dorgqr(
    LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]
  );
  timing::stop("QR");
}

define_method(
  void, qr_omm, (Hierarchical& A, Hierarchical& Q, Hierarchical& R)
) {
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == A.dim[1]);
  assert(R.dim[0] == A.dim[1]);
  assert(R.dim[1] == A.dim[1]);
  for (int64_t j=0; j<A.dim[1]; j++) {
    orthogonalize_block_col(j, A, Q, R(j, j));
    Hierarchical QjT(1, Q.dim[0]);
    for (int64_t i=0; i<Q.dim[0]; i++) {
      QjT(0, i) = transpose(Q(i, j));
    }
    for (int64_t k=j+1; k<A.dim[1]; k++) {
      for(int64_t i=0; i<A.dim[0]; i++) { //Rjk = Q*j^T x A*k
        gemm(QjT(0, i), A(i, k), R(j, k), 1, 1);
      }
      for(int64_t i=0; i<A.dim[0]; i++) { //A*k = A*k - Q*j x Rjk
        gemm(Q(i, j), R(j, k), A(i, k), -1, 1);
      }
    }
  }
}

define_method(void, qr_omm, (Matrix& A, Matrix& Q, Matrix& R)) {
  omm_error_handler("qr", {A, Q, R}, __FILE__, __LINE__);
  std::abort();
}


define_method(DensePair, make_left_orthogonal_omm, (const Dense& A)) {
  Dense Id(identity, std::vector<std::vector<double>>(), A.dim[0], A.dim[0]);
  return {std::move(Id), A};
}

define_method(DensePair, make_left_orthogonal_omm, (const LowRank& A)) {
  Dense Au(A.U);
  Dense Qu(get_n_rows(A.U), get_n_cols(A.U));
  Dense Ru(get_n_cols(A.U), get_n_cols(A.U));
  qr(Au, Qu, Ru);
  Dense RS(Ru.dim[0], A.S.dim[1]);
  gemm(Ru, A.S, RS, 1, 1);
  Dense RSV(RS.dim[0], get_n_cols(A.V));
  gemm(RS, A.V, RSV, 1, 1);
  return {std::move(Qu), std::move(RSV)};
}

define_method(DensePair, make_left_orthogonal_omm, (const Matrix& A)) {
  omm_error_handler("make_left_orthogonal", {A}, __FILE__, __LINE__);
  std::abort();
}


define_method(
  void, update_splitted_size_omm,
  (const Hierarchical& A, int64_t& rows, int64_t& cols)
) {
  rows += A.dim[0];
  cols = A.dim[1];
}

define_method(
  void, update_splitted_size_omm, (const Matrix&, int64_t& rows, int64_t&)
) {
  rows++;
}


define_method(
  MatrixProxy, split_by_column_omm,
  (const Dense& A, Hierarchical& storage, int64_t& currentRow)
) {
  Hierarchical splitted = split(A, 1, storage.dim[1], true);
  for(int64_t i=0; i<storage.dim[1]; i++)
    storage(currentRow, i) = splitted(0, i);
  currentRow++;
  return Dense(0, 0);
}

define_method(
  MatrixProxy, split_by_column_omm,
  (const LowRank& A, Hierarchical& storage, int64_t& currentRow)
) {
  LowRank _A(A);
  Dense Qu(get_n_rows(_A.U), get_n_cols(_A.U));
  Dense Ru(get_n_cols(_A.U), get_n_cols(_A.U));
  qr(_A.U, Qu, Ru);
  Dense RS = gemm(Ru, _A.S);
  Dense RSV = gemm(RS, _A.V);
  //Split R*S*V
  Hierarchical splitted = split(RSV, 1, storage.dim[1], true);
  for(int64_t i=0; i<storage.dim[1]; i++) {
    storage(currentRow, i) = splitted(0, i);
  }
  currentRow++;
  return std::move(Qu);
}

define_method(
  MatrixProxy, split_by_column_omm,
  (const Hierarchical& A, Hierarchical& storage, int64_t& currentRow)
) {
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<A.dim[1]; j++) {
      storage(currentRow, j) = A(i, j);
    }
    currentRow++;
  }
  return Dense(0, 0);
}

define_method(
  MatrixProxy, split_by_column_omm, (const Matrix& A, Matrix& storage, int64_t&)
) {
  omm_error_handler("split_by_column", {A, storage}, __FILE__, __LINE__);
  std::abort();
}


define_method(
  MatrixProxy, concat_columns_omm,
  (
    const Dense& A, const Hierarchical& splitted, const Dense&,
    int64_t& currentRow
  )
) {
  // In case of dense, combine the split dense matrices into one dense matrix
  Hierarchical SpCurRow(1, splitted.dim[1]);
  for(int64_t i=0; i<splitted.dim[1]; i++) {
    SpCurRow(0, i) = splitted(currentRow, i);
  }
  Dense concatenatedRow(SpCurRow);
  assert(A.dim[0] == concatenatedRow.dim[0]);
  assert(A.dim[1] == concatenatedRow.dim[1]);
  currentRow++;
  return std::move(concatenatedRow);
}

define_method(
  MatrixProxy, concat_columns_omm,
  (
    const LowRank& A, const Hierarchical& splitted, const Dense& Q,
    int64_t& currentRow
  )
) {
  // In case of lowrank, combine split dense matrices into single dense matrix
  // Then form a lowrank matrix with the stored Q
  Hierarchical SpCurRow(1, splitted.dim[1]);
  for(int64_t i=0; i<splitted.dim[1]; i++) {
    SpCurRow(0, i) = splitted(currentRow, i);
  }
  Dense concatenatedRow(SpCurRow);
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == A.rank);
  assert(concatenatedRow.dim[0] == A.rank);
  assert(concatenatedRow.dim[1] == A.dim[1]);
  LowRank _A(Dense(Q), Dense(identity, {}, A.rank, A.rank), concatenatedRow);
  currentRow++;
  return std::move(_A);
}

define_method(
  MatrixProxy, concat_columns_omm,
  (
    const Hierarchical& A, const Hierarchical& splitted, const Dense&,
    int64_t& currentRow
  )
  ) {
  //In case of hierarchical, just put element in respective cells
  assert(splitted.dim[1] == A.dim[1]);
  Hierarchical concatenatedRow(A.dim[0], A.dim[1]);
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<A.dim[1]; j++) {
      concatenatedRow(i, j) = splitted(currentRow, j);
    }
    currentRow++;
  }
  return std::move(concatenatedRow);
}

define_method(
  MatrixProxy, concat_columns_omm,
  (const Matrix& A, const Matrix& splitted, const Matrix& Q, int64_t&)
) {
  omm_error_handler("concat_columns", {A, splitted, Q}, __FILE__, __LINE__);
  std::abort();
}

void zero_lowtri(Matrix& A) {
  zero_lowtri_omm(A);
}

void zero_whole(Matrix& A) {
  zero_whole_omm(A);
}

define_method(void, zero_lowtri_omm, (Dense& A)) {
  for(int64_t i=0; i<A.dim[0]; i++)
    for(int64_t j=0; j<i; j++)
      A(i,j) = 0.0;
}

define_method(void, zero_lowtri_omm, (Matrix& A)) {
  omm_error_handler("zero_lowtri", {A}, __FILE__, __LINE__);
  std::abort();
}

define_method(void, zero_whole_omm, (Dense& A)) {
  A = 0.0;
}

define_method(void, zero_whole_omm, (LowRank& A)) {
  A.U = Dense(
    identity, std::vector<std::vector<double>>(),
    get_n_rows(A.U), get_n_cols(A.U)
  );
  A.S = 0.0;
  A.V = Dense(
    identity, std::vector<std::vector<double>>(),
    get_n_rows(A.V), get_n_cols(A.V)
  );
}

define_method(void, zero_whole_omm, (Matrix& A)) {
  omm_error_handler("zero_whole", {A}, __FILE__, __LINE__);
  std::abort();
}


std::tuple<Hierarchical, Hierarchical> split_block_col(
  int64_t j, const Hierarchical& A
) {
  int64_t splitRowSize = 0;
  int64_t splitColSize = 1;
  for(int64_t i=0; i<A.dim[0]; i++) {
    update_splitted_size(A(i, j), splitRowSize, splitColSize);
  }
  Hierarchical splitA(splitRowSize, splitColSize);
  Hierarchical QL(A.dim[0], 1);
  int64_t curRow = 0;
  for(int64_t i=0; i<A.dim[0]; i++) {
    QL(i, 0) = split_by_column(A(i, j), splitA, curRow);
  }
  return {std::move(splitA), std::move(QL)};
}

void restore_block_col(
  int64_t j,
  const Hierarchical& Q_splitA, const Hierarchical& QL, Hierarchical& Q
) {
  assert(QL.dim[0] == Q.dim[0]);
  int64_t curRow = 0;
  for(int64_t i=0; i<Q.dim[0]; i++) {
    Q(i, j) = concat_columns(Q(i, j), Q_splitA, QL(i, 0), curRow);
  }
}


define_method(
  void, orthogonalize_block_col_omm,
  (int64_t j, const Hierarchical& A, Hierarchical& Q, Dense& R)
) {
  Hierarchical Qu(A.dim[0], 1);
  Hierarchical B(A.dim[0], 1);
  for(int64_t i=0; i<A.dim[0]; i++) {
    std::tie(Qu(i, 0), B(i, 0)) = make_left_orthogonal(A(i, j));
  }
  Dense DB(B);
  Dense Qb(DB.dim[0], DB.dim[1]);
  Dense Rb(DB.dim[1], DB.dim[1]);
  qr(DB, Qb, Rb);
  R = std::move(Rb);
  //Slice Qb based on B
  Hierarchical HQb(B.dim[0], B.dim[1]);
  int64_t rowOffset = 0;
  for(int64_t i=0; i<HQb.dim[0]; i++) {
    int64_t dim_Bi[2]{get_n_rows(B(i, 0)), get_n_cols(B(i, 0))};
    Dense Qbi(dim_Bi[0], dim_Bi[1]);
    //add_copy_task(Qb, Qbi, rowOffset, 0);
    Qb.copy_to(Qbi, rowOffset);
    HQb(i, 0) = std::move(Qbi);
    rowOffset += dim_Bi[0];
  }
  for(int64_t i=0; i<A.dim[0]; i++) {
    gemm(Qu(i, 0), HQb(i, 0), Q(i, j), 1, 0);
  }
}

define_method(
  void, orthogonalize_block_col_omm,
  (int64_t j, const Hierarchical& A, Hierarchical& Q, Hierarchical& R)
) {
  Hierarchical splitA;
  Hierarchical QL;
  std::tie(splitA, QL) = split_block_col(j, A);
  Hierarchical Q_splitA(splitA);
  qr(splitA, Q_splitA, R);
  restore_block_col(j, Q_splitA, QL, Q);
}


define_method(Dense, get_right_factor_omm, (const Dense& A)) {
  return Dense(A);
}

define_method(Dense, get_right_factor_omm, (const LowRank& A)) {
  Dense SV = gemm(A.S, A.V);
  return std::move(SV);
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
    //add_copy_task(DRj, Rij, rowOffset, 0);
    DRj.copy_to(Rij, rowOffset);
    Rj(i, 0) = std::move(Rij);
    rowOffset += dim_Rij[0];
  }
  //Multiple block householder vectors with respective left factors
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
        Dense(identity, std::vector<std::vector<double>>(), get_n_rows(_C(0, 0)), get_n_rows(_C(0, 0))),
        _C(0, 0), A(k, j), -1, 1
      );
    }
    else { //Use gemm otherwise
      gemm(Y(i, k), C(0, 0), A(i, j), -1, 1);
    }
  }
}


void rq(Matrix& A, Matrix& R, Matrix& Q) { rq_omm(A, R, Q); }

define_method(void, rq_omm, (Dense& A, Dense& R, Dense& Q)) {
  assert(R.dim[0] == A.dim[0]);
  assert(R.dim[1] == A.dim[0]);
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == A.dim[1]);
  timing::start("DGERQF");
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<double> tau(k);
  LAPACKE_dgerqf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
  // TODO Consider making special function for this. Performance heavy and not
  // always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense derivative that remains in elementary reflector
  // form, uses dormqr instead of gemm and can be transformed to Dense via
  // dorgqr!
  for (int64_t i=0; i<R.dim[0]; i++) {
    for (int64_t j=i; j<R.dim[1]; j++) {
      R(i, j) = A(i, A.dim[1]-R.dim[1]+j);
    }
  }
  for(int64_t i=0; i<Q.dim[0]; i++) {
    for(int64_t j=0; j<std::min(A.dim[1]-R.dim[1]+i, Q.dim[1]); j++) {
      Q(i, j)= A(i, j);
    }
  }
  LAPACKE_dorgrq(
    LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]
  );
  timing::stop("DGERQF");
}

} // namespace hicma
