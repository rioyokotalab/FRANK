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

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"

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

bool need_split(const Matrix& A) {
  return need_split_omm(A);
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


define_method(void, qr_omm, (Dense& A, Dense& Q, Dense& R)) {
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == A.dim[1]);
  assert(R.dim[0] == A.dim[1]);
  assert(R.dim[1] == A.dim[1]);
  timing::start("QR");
  timing::start("DGEQRF");
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<double> tau(k);
  for(int64_t i=0; i<std::min(Q.dim[0], Q.dim[1]); i++) Q(i, i) = 1.0;
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
  timing::stop("DGEQRF");
  timing::start("DORGQR");
  // TODO Consider using A for the dorgqr and moving to Q afterwards! That
  // also simplify this loop.
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<A.dim[1]; j++) {
      if(j>=i)
        R(i, j) = A(i, j);
      else
        Q(i,j) = A(i,j);
    }
  }
  // TODO Consider making special function for this. Performance heavy
  // and not always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense deriative that remains in elementary
  // reflector form, uses dormqr instead of gemm and can be transformed to
  // Dense via dorgqr!
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q, Q.stride, &tau[0]);
  timing::stop("DORGQR");
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
    Hierarchical Qj(A.dim[0], 1);
    for (int64_t i = 0; i < A.dim[0]; i++) {
      Qj(i, 0) = Q(i, j);
    }
    Hierarchical Rjj(1, 1);
    Rjj(0, 0) = R(j, j);
    A.col_qr(j, Qj, Rjj);
    R(j, j) = Rjj(0, 0);
    for (int64_t i=0; i<A.dim[0]; i++) {
      Q(i, j) = Qj(i, 0);
    }
    Hierarchical TrQj(Qj);
    transpose(TrQj);
    for (int64_t k=j+1; k<A.dim[1]; k++) {
      //Take k-th column
      Hierarchical Ak(A.dim[0], 1);
      for (int64_t i=0; i<A.dim[0]; i++) {
        Ak(i, 0) = A(i, k);
      }
      Hierarchical Rjk(1, 1);
      Rjk(0, 0) = R(j, k);
      gemm(TrQj, Ak, Rjk, 1, 1); //Rjk = Q*j^T x A*k
      R(j, k) = Rjk(0, 0);
      gemm(Qj, Rjk, Ak, -1, 1); //A*k = A*k - Q*j x Rjk
      for (int64_t i=0; i<A.dim[0]; i++) {
        A(i, k) = Ak(i, 0);
      }
    }
  }
}

define_method(void, qr_omm, (Matrix& A, Matrix& Q, Matrix& R)) {
  omm_error_handler("qr", {A, Q, R}, __FILE__, __LINE__);
  std::abort();
}


define_method(bool, need_split_omm, ([[maybe_unused]] const Hierarchical& A)) {
  return true;
}

define_method(bool, need_split_omm, ([[maybe_unused]] const Matrix& A)) {
  return false;
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
  void, update_splitted_size_omm,
  (
    [[maybe_unused]] const Matrix& A,
    [[maybe_unused]] int64_t& rows, [[maybe_unused]] int64_t& cols
  )
) {
  rows++;
}


define_method(
  MatrixProxy, split_by_column_omm,
  (const Dense& A, Hierarchical& storage, int64_t& currentRow)
) {
  Hierarchical splitted(A, 1, storage.dim[1]);
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
  Hierarchical splitted(RSV, 1, storage.dim[1]);
  for(int64_t i=0; i<storage.dim[1]; i++) {
    storage(currentRow, i) = splitted(0, i);
  }
  currentRow++;
  return Qu;
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
  MatrixProxy, split_by_column_omm,
  (const Matrix& A, Matrix& storage, [[maybe_unused]] int64_t& currentRow)
) {
  omm_error_handler("split_by_column", {A, storage}, __FILE__, __LINE__);
  std::abort();
}


define_method(
  MatrixProxy, concat_columns_omm,
  (
    [[maybe_unused]] const Dense& A, const Hierarchical& splitted,
    [[maybe_unused]] const Dense& Q,
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
  return concatenatedRow;
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
  LowRank _A(A.dim[0], A.dim[1], A.rank);
  _A.U = Q;
  _A.V = concatenatedRow;
  _A.S = Dense(
    identity, std::vector<std::vector<double>>(), _A.rank, _A.rank);
  currentRow++;
  return _A;
}

define_method(
  MatrixProxy, concat_columns_omm,
  (
    const Hierarchical& A, const Hierarchical& splitted,
    [[maybe_unused]] const Dense& Q,
    int64_t& currentRow)
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
  return concatenatedRow;
}

define_method(
  MatrixProxy, concat_columns_omm,
  (
    const Matrix& A, const Matrix& splitted, const Matrix& Q,
    [[maybe_unused]] int64_t& currentRow
  )
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
    get_n_rows(A.V), get_n_cols(A.U)
  );
}

define_method(void, zero_whole_omm, (Matrix& A)) {
  omm_error_handler("zero_whole", {A}, __FILE__, __LINE__);
  std::abort();
}


void rq(Matrix& A, Matrix& R, Matrix& Q) { rq_omm(A, R, Q); }

define_method(void, rq_omm, (Dense& A, Dense& R, Dense& Q)) {
  assert(R.dim[0] == A.dim[0]);
  assert(R.dim[1] == A.dim[0]);
  assert(Q.dim[0] == A.dim[0]);
  assert(Q.dim[1] == A.dim[1]);
  timing::start("DGERQF");
  std::vector<double> tau(A.dim[1]);
  LAPACKE_dgerqf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
  // TODO Consider making special function for this. Performance heavy and not
  // always needed. If Q should be applied to something, use directly!
  // Alternatively, create Dense deriative that remains in elementary reflector
  // form, uses dormqr instead of gemm and can be transformed to Dense via
  // dorgqr!
  for (int64_t i=0; i<R.dim[0]; i++) {
    for (int64_t j=0; j<R.dim[1]; j++) {
      if (j>=i) R(i, j) = A(i, A.dim[1]-R.dim[1]+j);
    }
  }
  LAPACKE_dorgrq(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1], A.dim[0],
    &A, A.dim[1],
    &tau[0]
  );
  Q = std::move(A);
  timing::stop("DGERQF");
}

} // namespace hicma
