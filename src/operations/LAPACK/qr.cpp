#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc/transpose.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#include <cassert>
#include <tuple>
#include <utility>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"


namespace hicma
{

  void qr(Node& A, Node& Q, Node& R) {
    // TODO consider moving assertions here (same in other files)!
    qr_omm(A, Q, R);
  }

  bool need_split(const Node& A) {
    return need_split_omm(A);
  }

  std::tuple<Dense, Dense> make_left_orthogonal(const Node& A) {
    return make_left_orthogonal_omm(A);
  }

  void update_splitted_size(const Node& A, int& rows, int& cols) {
    update_splitted_size_omm(A, rows, cols);
  }

  NodeProxy split_by_column(const Node& A, Node& storage, int &currentRow) {
    return split_by_column_omm(A, storage, currentRow);
  }

  NodeProxy concat_columns(
    const Node& A, const Node& splitted, const Node& Q, int& currentRow
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
    int k = std::min(A.dim[0], A.dim[1]);
    std::vector<double> tau(k);
    for(int i=0; i<std::min(Q.dim[0], Q.dim[1]); i++) Q(i, i) = 1.0;
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
    timing::stop("DGEQRF");
    timing::start("DORGQR");
    // TODO Consider using A for the dorgqr and moving to Q afterwards! That
    // also simplify this loop.
    for(int i=0; i<A.dim[0]; i++) {
      for(int j=0; j<A.dim[1]; j++) {
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
    for (int j=0; j<A.dim[1]; j++) {
      Hierarchical Qj(A.dim[0], 1);
      for (int i = 0; i < A.dim[0]; i++) {
        Qj(i, 0) = Q(i, j);
      }
      Hierarchical Rjj(1, 1);
      Rjj(0, 0) = R(j, j);
      A.col_qr(j, Qj, Rjj);
      R(j, j) = Rjj(0, 0);
      for (int i=0; i<A.dim[0]; i++) {
        Q(i, j) = Qj(i, 0);
      }
      Hierarchical TrQj(Qj);
      transpose(TrQj);
      for (int k=j+1; k<A.dim[1]; k++) {
        //Take k-th column
        Hierarchical Ak(A.dim[0], 1);
        for (int i=0; i<A.dim[0]; i++) {
          Ak(i, 0) = A(i, k);
        }
        Hierarchical Rjk(1, 1);
        Rjk(0, 0) = R(j, k);
        gemm(TrQj, Ak, Rjk, 1, 1); //Rjk = Q*j^T x A*k
        R(j, k) = Rjk(0, 0);
        gemm(Qj, Rjk, Ak, -1, 1); //A*k = A*k - Q*j x Rjk
        for (int i=0; i<A.dim[0]; i++) {
          A(i, k) = Ak(i, 0);
        }
      }
    }
  }

  define_method(void, qr_omm, (Node& A, Node& Q, Node& R)) {
    omm_error_handler("qr", {A, Q, R}, __FILE__, __LINE__);
    abort();
  }


  define_method(
    bool, need_split_omm,
    ([[maybe_unused]] const Hierarchical& A)
  ) {
    return true;
  }

  define_method(bool, need_split_omm, ([[maybe_unused]] const Node& A)) {
    return false;
  }


  define_method(DensePair, make_left_orthogonal_omm, (const Dense& A)) {
    std::vector<double> x;
    Dense Id(identity, x, A.dim[0], A.dim[0]);
    return {std::move(Id), A};
  }

  define_method(DensePair, make_left_orthogonal_omm, (const LowRank& A)) {
    Dense Au(A.U());
    Dense Qu(A.U().dim[0], A.U().dim[1]);
    Dense Ru(A.U().dim[1], A.U().dim[1]);
    qr(Au, Qu, Ru);
    Dense RS(Ru.dim[0], A.S().dim[1]);
    gemm(Ru, A.S(), RS, 1, 1);
    Dense RSV(RS.dim[0], A.V().dim[1]);
    gemm(RS, A.V(), RSV, 1, 1);
    return {std::move(Qu), std::move(RSV)};
  }

  define_method(DensePair, make_left_orthogonal_omm, (const Node& A)) {
    omm_error_handler("make_left_orthogonal", {A}, __FILE__, __LINE__);
    abort();
  }


  define_method(
    void, update_splitted_size_omm,
    (const Hierarchical& A, int& rows, int& cols)
  ) {
    rows += A.dim[0];
    cols = A.dim[1];
  }

  define_method(
    void, update_splitted_size_omm, (const Node& A, int& rows, int& cols)
  ) {
    rows++;
  }


  define_method(
    NodeProxy, split_by_column_omm,
    (const Dense& A, Hierarchical& storage, int& currentRow)
  ) {
    Hierarchical splitted(A, 1, storage.dim[1]);
    for(int i=0; i<storage.dim[1]; i++)
      storage(currentRow, i) = splitted(0, i);
    currentRow++;
    return Dense(0, 0);
  }

  define_method(
    NodeProxy, split_by_column_omm,
    (const LowRank& A, Hierarchical& storage, int& currentRow)
  ) {
    LowRank _A(A);
    Dense Qu(_A.U().dim[0], _A.U().dim[1]);
    Dense Ru(_A.U().dim[1], _A.U().dim[1]);
    qr(_A.U(), Qu, Ru);
    Dense RS(Ru.dim[0], _A.S().dim[1]);
    gemm(Ru, _A.S(), RS, 1, 0);
    Dense RSV(RS.dim[0], _A.V().dim[1]);
    gemm(RS, _A.V(), RSV, 1, 0);
    //Split R*S*V
    Hierarchical splitted(RSV, 1, storage.dim[1]);
    for(int i=0; i<storage.dim[1]; i++) {
      storage(currentRow, i) = splitted(0, i);
    }
    currentRow++;
    return Qu;
  }

  define_method(
    NodeProxy, split_by_column_omm,
    (const Hierarchical& A, Hierarchical& storage, int& currentRow)
  ) {
    for(int i=0; i<A.dim[0]; i++) {
      for(int j=0; j<A.dim[1]; j++) {
        storage(currentRow, j) = A(i, j);
      }
      currentRow++;
    }
    return Dense(0, 0);
  }

  define_method(
    NodeProxy, split_by_column_omm,
    (const Node& A, Node& storage, [[maybe_unused]] int& currentRow)
  ) {
    omm_error_handler("split_by_column", {A, storage}, __FILE__, __LINE__);
    abort();
  }


  define_method(
    NodeProxy, concat_columns_omm,
    (const Dense& A, const Hierarchical& splitted, const Dense& Q, int& currentRow)
  ) {
    //In case of dense, combine the spllited dense matrices into one dense matrix
    Hierarchical SpCurRow(1, splitted.dim[1]);
    for(int i=0; i<splitted.dim[1]; i++) {
      SpCurRow(0, i) = splitted(currentRow, i);
    }
    Dense concatenatedRow(SpCurRow);
    assert(A.dim[0] == concatenatedRow.dim[0]);
    assert(A.dim[1] == concatenatedRow.dim[1]);
    currentRow++;
    return concatenatedRow;
  }

  define_method(
    NodeProxy, concat_columns_omm,
    (const LowRank& A, const Hierarchical& splitted, const Dense& Q, int& currentRow)
  ) {
    //In case of lowrank, combine splitted dense matrices into single dense matrix
    //Then form a lowrank matrix with the stored Q
    std::vector<double> x;
    Hierarchical SpCurRow(1, splitted.dim[1]);
    for(int i=0; i<splitted.dim[1]; i++) {
      SpCurRow(0, i) = splitted(currentRow, i);
    }
    Dense concatenatedRow(SpCurRow);
    assert(Q.dim[0] == A.dim[0]);
    assert(Q.dim[1] == A.rank);
    assert(concatenatedRow.dim[0] == A.rank);
    assert(concatenatedRow.dim[1] == A.dim[1]);
    LowRank _A(A.dim[0], A.dim[1], A.rank);
    _A.U() = Q;
    _A.V() = concatenatedRow;
    _A.S() = Dense(identity, x, _A.rank, _A.rank);
    currentRow++;
    return _A;
  }

  define_method(
    NodeProxy, concat_columns_omm,
    (
      const Hierarchical& A, const Hierarchical& splitted, const Dense& Q,
      int& currentRow)
    ) {
    //In case of hierarchical, just put element in respective cells
    assert(splitted.dim[1] == A.dim[1]);
    Hierarchical concatenatedRow(A.dim[0], A.dim[1]);
    for(int i=0; i<A.dim[0]; i++) {
      for(int j=0; j<A.dim[1]; j++) {
        concatenatedRow(i, j) = splitted(currentRow, j);
      }
      currentRow++;
    }
    return concatenatedRow;
  }

  define_method(
    NodeProxy, concat_columns_omm,
    (
      const Node& A, const Node& splitted, const Node& Q,
      [[maybe_unused]] int& currentRow)
    ) {
    omm_error_handler("concat_columns", {A, splitted, Q}, __FILE__, __LINE__);
    abort();
  }


  void zero_lowtri(Node& A) {
    zero_lowtri_omm(A);
  }

  void zero_whole(Node& A) {
    zero_whole_omm(A);
  }

  define_method(void, zero_lowtri_omm, (Dense& A)) {
    for(int i=0; i<A.dim[0]; i++)
      for(int j=0; j<i; j++)
        A(i,j) = 0.0;
  }

  define_method(void, zero_lowtri_omm, (Node& A)) {
    omm_error_handler("zero_lowtri", {A}, __FILE__, __LINE__);
    abort();
  }

  define_method(void, zero_whole_omm, (Dense& A)) {
    A = 0.0;
  }

  define_method(void, zero_whole_omm, (LowRank& A)) {
    A.U() = 0.0;
    for(int i=0; i<std::min(A.U().dim[0], A.U().dim[1]); i++) A.U()(i, i) = 1.0;
    A.S() = 0.0;
    A.V() = 0.0;
    for(int i=0; i<std::min(A.V().dim[0], A.V().dim[1]); i++) A.V()(i, i) = 1.0;
  }

  define_method(void, zero_whole_omm, (Node& A)) {
    omm_error_handler("zero_whole", {A}, __FILE__, __LINE__);
    abort();
  }


  void rq(Node& A, Node& R, Node& Q) {
    rq_omm(A, R, Q);
  }

  define_method(void, rq_omm, (Dense& A, Dense& R, Dense& Q)) {
    assert(R.dim[0] == A.dim[0]);
    assert(R.dim[1] == A.dim[0]);
    assert(Q.dim[0] == A.dim[0]);
    assert(Q.dim[1] == A.dim[1]);
    timing::start("DGERQF");
    std::vector<double> tau(A.dim[1]);
    LAPACKE_dgerqf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A, A.stride, &tau[0]);
    // TODO Consider making special function for this. Performance heavy
    // and not always needed. If Q should be applied to something, use directly!
    // Alternatively, create Dense deriative that remains in elementary
    // reflector form, uses dormqr instead of gemm and can be transformed to
    // Dense via dorgqr!
    for (int i=0; i<R.dim[0]; i++) {
      for (int j=0; j<R.dim[1]; j++) {
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
