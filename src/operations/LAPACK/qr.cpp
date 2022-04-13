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
// explicit template initialization (these are the only available types)
template void triangularize_block_col(int64_t, Hierarchical<float>&, Hierarchical<float>&);
template void triangularize_block_col(int64_t, Hierarchical<double>&, Hierarchical<double>&);
template void apply_block_col_householder(const Hierarchical<float>&, const Hierarchical<float>&, int64_t, bool, Hierarchical<float>&, int64_t);
template void apply_block_col_householder(const Hierarchical<double>&, const Hierarchical<double>&, int64_t, bool, Hierarchical<double>&, int64_t);

void qr(Matrix& A, Matrix& Q, Matrix& R) {
  // TODO consider moving assertions here (same in other files)!
  // TODO hierarchical and dense have different assertions check which ones
  qr_omm(A, Q, R);
}

template<typename T>
std::tuple<Dense<T>, Dense<T>> make_left_orthogonal(const Matrix& A) {
  return make_left_orthogonal_omm(A);
}

template<typename T>
MatrixPair dense_make_left_orthogonal(const Dense<T>& A) {
  Dense<T> Id(identity, A.dim[0], A.dim[0]);
  MatrixPair out {std::move(Id), A};
  return out;
}

define_method(MatrixPair, make_left_orthogonal_omm, (const Dense<float>& A)) {
  return dense_make_left_orthogonal(A);
}

define_method(MatrixPair, make_left_orthogonal_omm, (const Dense<double>& A)) {
  return dense_make_left_orthogonal(A);
}

template<typename T>
MatrixPair low_rank_make_left_orthogonal(const LowRank<T>& A) {
  Dense<T> Au(A.U);
  Dense<T> Qu(get_n_rows(A.U), get_n_cols(A.U));
  Dense<T> Ru(get_n_cols(A.U), get_n_cols(A.U));
  qr(Au, Qu, Ru);
  Dense<T> RS(Ru.dim[0], A.S.dim[1]);
  gemm(Ru, A.S, RS, 1, 1);
  Dense<T> RSV(RS.dim[0], get_n_cols(A.V));
  gemm(RS, A.V, RSV, 1, 1);
  return {std::move(Qu), std::move(RSV)};
}

define_method(MatrixPair, make_left_orthogonal_omm, (const LowRank<float>& A)) {
  return low_rank_make_left_orthogonal(A);
}

define_method(MatrixPair, make_left_orthogonal_omm, (const LowRank<double>& A)) {
  return low_rank_make_left_orthogonal(A);
}

define_method(MatrixPair, make_left_orthogonal_omm, (const Matrix& A)) {
  omm_error_handler("make_left_orthogonal", {A}, __FILE__, __LINE__);
  std::abort();
}

void update_splitted_size(const Matrix& A, int64_t& rows, int64_t& cols) {
  update_splitted_size_omm(A, rows, cols);
}

template<typename T>
void hierarchical_update_splitted_size(const Hierarchical<T>& A, int64_t& rows, int64_t& cols) {
  rows += A.dim[0];
  cols = A.dim[1];
}

define_method(
  void, update_splitted_size_omm,
  (const Hierarchical<float>& A, int64_t& rows, int64_t& cols)
) {
  hierarchical_update_splitted_size(A, rows, cols);
}

define_method(
  void, update_splitted_size_omm,
  (const Hierarchical<double>& A, int64_t& rows, int64_t& cols)
) {
  hierarchical_update_splitted_size(A, rows, cols);
}

define_method(
  void, update_splitted_size_omm, (const Matrix&, int64_t& rows, int64_t&)
) {
  rows++;
}

MatrixProxy split_by_column(
  const Matrix& A, Matrix& storage, int64_t &currentRow
) {
  return split_by_column_omm(A, storage, currentRow);
}

template<typename T>
MatrixProxy split_by_column_d_h(const Dense<T>& A, Hierarchical<T>& storage, int64_t& currentRow) {
  Hierarchical<T> splitted = split<T>(A, 1, storage.dim[1], true);
  for(int64_t i=0; i<storage.dim[1]; i++)
    storage(currentRow, i) = splitted(0, i);
  currentRow++;
  return Dense<T>((int64_t)0, 0);
}

define_method(
  MatrixProxy, split_by_column_omm,
  (const Dense<float>& A, Hierarchical<float>& storage, int64_t& currentRow)
) {
  return split_by_column_d_h(A, storage, currentRow);
}

define_method(
  MatrixProxy, split_by_column_omm,
  (const Dense<double>& A, Hierarchical<double>& storage, int64_t& currentRow)
) {
  return split_by_column_d_h(A, storage, currentRow);
}

template<typename T>
MatrixProxy split_by_column_lr_h(const LowRank<T>& A, Hierarchical<T>& storage, int64_t& currentRow) {
  LowRank<T> _A(A);
  Dense<T> Qu(get_n_rows(_A.U), get_n_cols(_A.U));
  Dense<T> Ru(get_n_cols(_A.U), get_n_cols(_A.U));
  qr(_A.U, Qu, Ru);
  Dense<T> RS = gemm(Ru, _A.S);
  Dense<T> RSV = gemm(RS, _A.V);
  //Split R*S*V
  Hierarchical<T> splitted = split<T>(RSV, 1, storage.dim[1], true);
  for(int64_t i=0; i<storage.dim[1]; i++) {
    storage(currentRow, i) = splitted(0, i);
  }
  currentRow++;
  return std::move(Qu);
}

define_method(
  MatrixProxy, split_by_column_omm,
  (const LowRank<float>& A, Hierarchical<float>& storage, int64_t& currentRow)
) {
  return split_by_column_lr_h(A, storage, currentRow);
}

define_method(
  MatrixProxy, split_by_column_omm,
  (const LowRank<double>& A, Hierarchical<double>& storage, int64_t& currentRow)
) {
  return split_by_column_lr_h(A, storage, currentRow);
}

template<typename T>
MatrixProxy split_by_column_h_h(const Hierarchical<T>& A, Hierarchical<T>& storage, int64_t& currentRow) {
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<A.dim[1]; j++) {
      storage(currentRow, j) = A(i, j);
    }
    currentRow++;
  }
  return Dense<T>((int64_t)0, 0);
}

define_method(
  MatrixProxy, split_by_column_omm,
  (const Hierarchical<float>& A, Hierarchical<float>& storage, int64_t& currentRow)
) {
  return split_by_column_h_h(A, storage, currentRow);
}

define_method(
  MatrixProxy, split_by_column_omm,
  (const Hierarchical<double>& A, Hierarchical<double>& storage, int64_t& currentRow)
) {
  return split_by_column_h_h(A, storage, currentRow);
}

define_method(
  MatrixProxy, split_by_column_omm, (const Matrix& A, Matrix& storage, int64_t&)
) {
  omm_error_handler("split_by_column", {A, storage}, __FILE__, __LINE__);
  std::abort();
}

MatrixProxy concat_columns(
  const Matrix& A, const Matrix& splitted, const Matrix& Q, int64_t& currentRow
) {
  return concat_columns_omm(A, splitted, Q, currentRow);
}

template<typename T>
MatrixProxy concat_columns_h(const Hierarchical<T>& splitted, int64_t currentRow) {
  // In case of dense, combine the split dense matrices into one dense matrix
  Hierarchical<T> SpCurRow(1, splitted.dim[1]);
  for(int64_t i=0; i<splitted.dim[1]; i++) {
    SpCurRow(0, i) = splitted(currentRow, i);
  }
  Dense<T> concatenatedRow(SpCurRow);
  assert(A.dim[0] == concatenatedRow.dim[0]);
  assert(A.dim[1] == concatenatedRow.dim[1]);
  currentRow++;
  return std::move(concatenatedRow);
}

define_method(
  MatrixProxy, concat_columns_omm,
  (
    const Dense<float>&, const Hierarchical<float>& splitted, const Dense<float>&,
    int64_t& currentRow
  )
) {
  return concat_columns_h(splitted, currentRow);
}

define_method(
  MatrixProxy, concat_columns_omm,
  (
    const Dense<double>&, const Hierarchical<double>& splitted, const Dense<double>&,
    int64_t& currentRow
  )
) {
  return concat_columns_h(splitted, currentRow);
}

template<typename T>
MatrixProxy concat_columns_lr_h_d(const LowRank<T>& A, const Hierarchical<T>& splitted, const Dense<T>& Q, int64_t currentRow) {
  // In case of lowrank, combine split dense matrices into single dense matrix
  // Then form a lowrank matrix with the stored Q
  Hierarchical<T> SpCurRow(1, splitted.dim[1]);
  for(int64_t i=0; i<splitted.dim[1]; i++) {
    SpCurRow(0, i) = splitted(currentRow, i);
  }
  Dense<T> concatenatedRow(SpCurRow);
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == A.rank);
  assert(concatenatedRow.dim[0] == A.rank);
  assert(concatenatedRow.dim[1] == A.dim[1]);
  LowRank<T> _A(Dense<T>(Q), Dense<T>(identity, A.rank, A.rank), concatenatedRow);
  currentRow++;
  return std::move(_A);
}

define_method(
  MatrixProxy, concat_columns_omm,
  (
    const LowRank<float>& A, const Hierarchical<float>& splitted, const Dense<float>& Q,
    int64_t& currentRow
  )
) {
  return concat_columns_lr_h_d(A, splitted, Q, currentRow);
}

define_method(
  MatrixProxy, concat_columns_omm,
  (
    const LowRank<double>& A, const Hierarchical<double>& splitted, const Dense<double>& Q,
    int64_t& currentRow
  )
) {
  return concat_columns_lr_h_d(A, splitted, Q, currentRow);
}

template<typename T>
MatrixProxy concat_columns_h_h(const Hierarchical<T>& A, const Hierarchical<T>& splitted, int64_t currentRow) {
  //In case of hierarchical, just put element in respective cells
  assert(splitted.dim[1] == A.dim[1]);
  Hierarchical<T> concatenatedRow(A.dim[0], A.dim[1]);
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
  (
    const Hierarchical<float>& A, const Hierarchical<float>& splitted, const Dense<float>&,
    int64_t& currentRow
  )
  ) {
  return concat_columns_h_h(A, splitted, currentRow);
}

define_method(
  MatrixProxy, concat_columns_omm,
  (
    const Hierarchical<double>& A, const Hierarchical<double>& splitted, const Dense<double>&,
    int64_t& currentRow
  )
  ) {
  return concat_columns_h_h(A, splitted, currentRow);
}

define_method(
  MatrixProxy, concat_columns_omm,
  (const Matrix& A, const Matrix& splitted, const Matrix& Q, int64_t&)
) {
  omm_error_handler("concat_columns", {A, splitted, Q}, __FILE__, __LINE__);
  std::abort();
}

template<typename T>
std::tuple<Hierarchical<T>, Hierarchical<T>> split_block_col(
  int64_t j, const Hierarchical<T>& A
) {
  int64_t splitRowSize = 0;
  int64_t splitColSize = 1;
  for(int64_t i=0; i<A.dim[0]; i++) {
    update_splitted_size(A(i, j), splitRowSize, splitColSize);
  }
  Hierarchical<T> splitA(splitRowSize, splitColSize);
  Hierarchical<T> QL(A.dim[0], 1);
  int64_t curRow = 0;
  for(int64_t i=0; i<A.dim[0]; i++) {
    QL(i, 0) = split_by_column(A(i, j), splitA, curRow);
  }
  return {std::move(splitA), std::move(QL)};
}

template<typename T>
void restore_block_col(
  int64_t j,
  const Hierarchical<T>& Q_splitA, const Hierarchical<T>& QL, Hierarchical<T>& Q
) {
  assert(QL.dim[0] == Q.dim[0]);
  int64_t curRow = 0;
  for(int64_t i=0; i<Q.dim[0]; i++) {
    Q(i, j) = concat_columns(Q(i, j), Q_splitA, QL(i, 0), curRow);
  }
}

void orthogonalize_block_col(int64_t j, const Matrix& A, Matrix& Q, Matrix& R) {
  orthogonalize_block_col_omm(j, A, Q, R);
}

template<typename T>
void orthogonalize_block_col_h_h_d(int64_t j, const Hierarchical<T>&A, Hierarchical<T>& Q, Dense<T>& R) {
  Hierarchical<T> Qu(A.dim[0], 1);
  Hierarchical<T> B(A.dim[0], 1);
  for(int64_t i=0; i<A.dim[0]; i++) {
    std::tie(Qu(i, 0), B(i, 0)) = make_left_orthogonal<T>(A(i, j));
  }
  Dense<T> Qb(B);
  Dense<T> Rb(Qb.dim[1], Qb.dim[1]);
  mgs_qr(Qb, Rb);
  R = std::move(Rb);
  //Slice Qb based on B
  Hierarchical<T> HQb(B.dim[0], B.dim[1]);
  int64_t rowOffset = 0;
  for(int64_t i=0; i<HQb.dim[0]; i++) {
    int64_t dim_Bi[2]{get_n_rows(B(i, 0)), get_n_cols(B(i, 0))};
    Dense<T> Qbi(dim_Bi[0], dim_Bi[1]);
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
  (int64_t j, const Hierarchical<float>& A, Hierarchical<float>& Q, Dense<float>& R)
) {
  orthogonalize_block_col_h_h_d(j, A, Q, R);
}

define_method(
  void, orthogonalize_block_col_omm,
  (int64_t j, const Hierarchical<double>& A, Hierarchical<double>& Q, Dense<double>& R)
) {
  orthogonalize_block_col_h_h_d(j, A, Q, R);
}

template<typename T>
void orthogonalize_block_col_h_h_h(int64_t j, const Hierarchical<T>&A, Hierarchical<T>& Q, Hierarchical<T>& R) {
  Hierarchical<T> splitA;
  Hierarchical<T> QL;
  std::tie(splitA, QL) = split_block_col(j, A);
  Hierarchical<T> Q_splitA(splitA);
  qr(splitA, Q_splitA, R);
  restore_block_col(j, Q_splitA, QL, Q);
}

define_method(
  void, orthogonalize_block_col_omm,
  (int64_t j, const Hierarchical<float>& A, Hierarchical<float>& Q, Hierarchical<float>& R)
) {
  orthogonalize_block_col_h_h_h(j, A, Q, R);
}

define_method(
  void, orthogonalize_block_col_omm,
  (int64_t j, const Hierarchical<double>& A, Hierarchical<double>& Q, Hierarchical<double>& R)
) {
  orthogonalize_block_col_h_h_h(j, A, Q, R);
}

template<typename T>
Dense<T> get_right_factor(const Matrix& A) {
  return get_right_factor_omm(A);
}

define_method(MatrixProxy, get_right_factor_omm, (const Dense<float>& A)) {
  return Dense<float>(A);
}

define_method(MatrixProxy, get_right_factor_omm, (const Dense<double>& A)) {
  return Dense<double>(A);
}

define_method(MatrixProxy, get_right_factor_omm, (const LowRank<float>& A)) {
  Dense<float> SV = gemm(A.S, A.V);
  return std::move(SV);
}

define_method(MatrixProxy, get_right_factor_omm, (const LowRank<double>& A)) {
  Dense<double> SV = gemm(A.S, A.V);
  return std::move(SV);
}

void update_right_factor(Matrix& A, Matrix& R) {
  update_right_factor_omm(A, R);
}

define_method(
  void, update_right_factor_omm,
  (Dense<float>& A, Dense<float>& R)
) {
  A = std::move(R);
}

define_method(
  void, update_right_factor_omm,
  (Dense<double>& A, Dense<double>& R)
) {
  A = std::move(R);
}

template<typename T>
void update_right_factor_lr(LowRank<T>& A, Dense<T>& R) {
  A.S = 0.0;
  for(int64_t i=0; i<std::min(A.S.dim[0], A.S.dim[1]); i++) {
    A.S(i, i) = 1.0;
  }
  A.V = std::move(R);
}

define_method(
  void, update_right_factor_omm,
  (LowRank<float>& A, Dense<float>& R)
) {
  update_right_factor_lr(A, R);
}

define_method(
  void, update_right_factor_omm,
  (LowRank<double>& A, Dense<double>& R)
) {
  update_right_factor_lr(A, R);
}

//single precision
define_method(void, qr_omm, (Dense<float>& A, Dense<float>& Q, Dense<float>& R)) {
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == R.dim[0]);
  assert(R.dim[1] == A.dim[1]);
  timing::start("SQR");
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<float> tau(k);
  LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
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
  LAPACKE_sorgqr(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]);
  timing::stop("SQR");
}

// double precision
define_method(void, qr_omm, (Dense<double>& A, Dense<double>& Q, Dense<double>& R)) {
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == R.dim[0]);
  assert(R.dim[1] == A.dim[1]);
  timing::start("DQR");
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
  timing::stop("DQR");
}

template<typename T>
void hierarchical_qr(Hierarchical<T>& A, Hierarchical<T>& Q, Hierarchical<T>& R) {
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == A.dim[1]);
  assert(R.dim[0] == A.dim[1]);
  assert(R.dim[1] == A.dim[1]);
  for (int64_t j=0; j<A.dim[1]; j++) {
    orthogonalize_block_col(j, A, Q, R(j, j));
    Hierarchical<T> QjT(1, Q.dim[0]);
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

define_method(
  void, qr_omm, (Hierarchical<float>& A, Hierarchical<float>& Q, Hierarchical<float>& R)
) {
  hierarchical_qr(A, Q, R);
}

define_method(
  void, qr_omm, (Hierarchical<double>& A, Hierarchical<double>& Q, Hierarchical<double>& R)
) {
  hierarchical_qr(A, Q, R);
}

define_method(void, qr_omm, (Matrix& A, Matrix& Q, Matrix& R)) {
  omm_error_handler("qr", {A, Q, R}, __FILE__, __LINE__);
  std::abort();
}

void zero_lowtri(Matrix& A) {
  zero_lowtri_omm(A);
}

template<typename T>
void dense_zero_lowtri(Dense<T>& A) {
  for(int64_t i=0; i<A.dim[0]; i++)
    for(int64_t j=0; j<i; j++)
      A(i,j) = 0.0;
}

define_method(void, zero_lowtri_omm, (Dense<float>& A)) {
 dense_zero_lowtri(A);
}

define_method(void, zero_lowtri_omm, (Dense<double>& A)) {
 dense_zero_lowtri(A);
}

define_method(void, zero_lowtri_omm, (Matrix& A)) {
  omm_error_handler("zero_lowtri", {A}, __FILE__, __LINE__);
  std::abort();
}

void zero_whole(Matrix& A) {
  zero_whole_omm(A);
}

template<typename T>
void dense_zero_whole(Dense<T>& A) {
  A = 0.0;
}

define_method(void, zero_whole_omm, (Dense<float>& A)) {
  dense_zero_whole(A);
}

define_method(void, zero_whole_omm, (Dense<double>& A)) {
  dense_zero_whole(A);
}

template<typename T>
void low_rank_zero_whole(LowRank<T>& A) {
  A.U = Dense<T>(
    identity,
    get_n_rows(A.U), get_n_cols(A.U)
  );
  A.S = 0.0;
  A.V = Dense<T>(
    identity,
    get_n_rows(A.V), get_n_cols(A.V)
  );
}

define_method(void, zero_whole_omm, (LowRank<float>& A)) {
  low_rank_zero_whole(A);
}

define_method(void, zero_whole_omm, (LowRank<double>& A)) {
  low_rank_zero_whole(A);
}

define_method(void, zero_whole_omm, (Matrix& A)) {
  omm_error_handler("zero_whole", {A}, __FILE__, __LINE__);
  std::abort();
}

template<typename T>
void triangularize_block_col(int64_t j, Hierarchical<T>& A, Hierarchical<T>& S) {
  //Put right factors of Aj into Rj
  Hierarchical<T> Rj(A.dim[0]-j, 1);
  for(int64_t i=0; i<Rj.dim[0]; i++) {
    Rj(i, 0) = get_right_factor<T>(A(j+i, j));
  }
  //QR on concatenated right factors
  Dense<T> DRj(Rj);
  Dense<T> Sj(DRj.dim[1], DRj.dim[1]);
  geqrt(DRj, Sj);
  S(j, 0) = std::move(Sj);
  //Slice DRj based on Rj
  int64_t rowOffset = 0;
  for(int64_t i=0; i<Rj.dim[0]; i++) {
    assert(DRj.dim[1] == get_n_cols(Rj(i, 0)));
    int64_t dim_Rij[2]{get_n_rows(Rj(i, 0)), get_n_cols(Rj(i, 0))};
    Dense<T> Rij(dim_Rij[0], dim_Rij[1]);
    DRj.copy_to(Rij, rowOffset);
    Rj(i, 0) = std::move(Rij);
    rowOffset += dim_Rij[0];
  }
  //Multiple block householder vectors with respective left factors
  for(int64_t i=0; i<Rj.dim[0]; i++) {
    update_right_factor(A(j+i, j), Rj(i, 0));
  }
}

template<typename T>
void apply_block_col_householder(const Hierarchical<T>& Y, const Hierarchical<T>& S, int64_t k, bool trans, Hierarchical<T>& A, int64_t j) {
  assert(A.dim[0] == Y.dim[0]);
  Hierarchical<T> YkT(1, Y.dim[0]-k);
  for(int64_t i=0; i<YkT.dim[1]; i++)
    YkT(0, i) = transpose(Y(i+k,k));

  Hierarchical<T> C(1, 1);
  C(0, 0) = A(k, j); //C = Akj
  trmm(Y(k, k), C(0, 0), 'l', 'l', 't', 'u', 1); //C = Ykk^T x Akj
  for(int64_t i=k+1; i<A.dim[0]; i++) {
    gemm(YkT(0, i-k), A(i, j), C(0, 0), 1, 1); //C += Yik^T x Aij
  }
  trmm(S(k, 0), C(0, 0), 'l', 'u', trans ? 't' : 'n', 'n', 1); //C = (S or S^T) x C
  for(int64_t i=k; i<A.dim[0]; i++) {
    //Aij = Aij - Yik x C
    if(i == k) { //Use trmm since Ykk is unit lower triangular
      Hierarchical<T> _C(C);
      trmm(Y(k, k), _C(0, 0), 'l', 'l', 'n', 'u', 1);
      gemm(
        Dense<T>(identity, get_n_rows(_C(0, 0)), get_n_rows(_C(0, 0))),
        _C(0, 0), A(k, j), -1, 1
      );
    }
    else { //Use gemm otherwise
      gemm(Y(i, k), C(0, 0), A(i, j), -1, 1);
    }
  }
}

void rq(Matrix& A, Matrix& R, Matrix& Q) { rq_omm(A, R, Q); }

// single precision
define_method(void, rq_omm, (Dense<float>& A, Dense<float>& R, Dense<float>& Q)) {
  assert(R.dim[0] == A.dim[0]);
  assert(R.dim[1] == Q.dim[0]);
  assert(Q.dim[1] == A.dim[1]);
  timing::start("SRQ");
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<float> tau(k);
  LAPACKE_sgerqf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
  // Copy upper triangular into R
  for(int64_t i=0; i<R.dim[0]; i++) {
    for(int64_t j=std::max(i+A.dim[1]-A.dim[0], (int64_t)0); j<A.dim[1]; j++) {
      R(i, j+R.dim[1]-A.dim[1]) = A(i, j);
    }
  }
  // Copy strictly lower part into Q
  for(int64_t i=std::max(A.dim[0]-A.dim[1], (int64_t)0); i<A.dim[0]; i++) {
    for(int64_t j=0; j<(i+A.dim[1]-A.dim[0]); j++) {
      Q(i+Q.dim[0]-A.dim[0], j) = A(i, j);
    }
  }
  // TODO Consider making special function for this. Performance heavy and not
  // always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense derivative that remains in elementary reflector
  // form, uses dormrq instead of gemm and can be transformed to Dense via
  // dorgrq!
  LAPACKE_sorgrq(
    LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]
  );
  timing::stop("SRQ");
}

// double precision
define_method(void, rq_omm, (Dense<double>& A, Dense<double>& R, Dense<double>& Q)) {
  assert(R.dim[0] == A.dim[0]);
  assert(R.dim[1] == Q.dim[0]);
  assert(Q.dim[1] == A.dim[1]);
  timing::start("DRQ");
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<double> tau(k);
  LAPACKE_dgerqf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
  // Copy upper triangular into R
  for(int64_t i=0; i<R.dim[0]; i++) {
    for(int64_t j=std::max(i+A.dim[1]-A.dim[0], (int64_t)0); j<A.dim[1]; j++) {
      R(i, j+R.dim[1]-A.dim[1]) = A(i, j);
    }
  }
  // Copy strictly lower part into Q
  for(int64_t i=std::max(A.dim[0]-A.dim[1], (int64_t)0); i<A.dim[0]; i++) {
    for(int64_t j=0; j<(i+A.dim[1]-A.dim[0]); j++) {
      Q(i+Q.dim[0]-A.dim[0], j) = A(i, j);
    }
  }
  // TODO Consider making special function for this. Performance heavy and not
  // always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense derivative that remains in elementary reflector
  // form, uses dormrq instead of gemm and can be transformed to Dense via
  // dorgrq!
  LAPACKE_dorgrq(
    LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]
  );
  timing::stop("DRQ");
}

template<>
void mgs_qr(Dense<float>& A, Dense<float>& R) {
  assert(A.dim[1] == R.dim[0]);
  assert(A.dim[1] == R.dim[1]);
  for(int j = 0; j < A.dim[1]; j++) {
    R(j, j) = LAPACKE_slange(LAPACK_ROW_MAJOR, 'F',
			     A.dim[0], 1, &A + j, A.dim[1]);
    double alpha = 1./R(j, j);
    cblas_sscal(A.dim[0], alpha, &A + j, A.dim[1]);
    for(int k = j + 1; k < A.dim[1]; k++) {
      R(j, k) = cblas_sdot(A.dim[0], &A + j, A.dim[1],
			   &A + k, A.dim[1]);
      cblas_saxpy(A.dim[0], -R(j, k),
		  &A + j, A.dim[1], &A + k, A.dim[1]);
    }
  }
}

template<>
void mgs_qr(Dense<double>& A, Dense<double>& R) {
  assert(A.dim[1] == R.dim[0]);
  assert(A.dim[1] == R.dim[1]);
  for(int j = 0; j < A.dim[1]; j++) {
    R(j, j) = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'F',
			     A.dim[0], 1, &A + j, A.dim[1]);
    double alpha = 1./R(j, j);
    cblas_dscal(A.dim[0], alpha, &A + j, A.dim[1]);
    for(int k = j + 1; k < A.dim[1]; k++) {
      R(j, k) = cblas_ddot(A.dim[0], &A + j, A.dim[1],
			   &A + k, A.dim[1]);
      cblas_daxpy(A.dim[0], -R(j, k),
		  &A + j, A.dim[1], &A + k, A.dim[1]);
    }
  }
}

} // namespace hicma
