#include "hicma/operations/qr.h"

#include "hicma/node.h"
#include "hicma/node_proxy.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations/gemm.h"
#include "hicma/operations/transpose.h"

#include <cassert>
#include <iostream>
#include <tuple>
#include <utility>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

  void qr(Node& A, Node& Q, Node& R) {
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

  NodeProxy concat_columns(const Node& A, const Node& splitted, int& currentRow, const Node& Q) {
    return concat_columns_omm(A, splitted, currentRow, Q);
  }

  void zero_lowtri(Node& A) {
    zero_lowtri_omm(A);
  }

  void zero_whole(Node& A) {
    zero_whole_omm(A);
  }

  BEGIN_SPECIALIZATION(qr_omm, void, Dense& A, Dense& Q, Dense& R) {
    int k = std::min(A.dim[0], A.dim[1]);
    std::vector<double> tau(k);
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A[0], A.dim[1], &tau[0]);
    for(int i=0; i<std::min(Q.dim[0], Q.dim[1]); i++) Q(i, i) = 1.0;
    for(int i=0; i<A.dim[0]; i++) {
      for(int j=0; j<A.dim[1]; j++) {
        if(j>=i)
          R[i*(A.dim[1])+j] = A[i*(A.dim[1])+j];
        else
          Q(i,j) = A(i,j);
      }
    }
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Q.dim[0], Q.dim[1], k, &Q[0], Q.dim[1], &tau[0]);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(qr_omm, void, Hierarchical& A, Hierarchical& Q, Hierarchical& R) {
    assert(Q.dim[0] == A.dim[0]);
    assert(Q.dim[1] == A.dim[1]);
    assert(R.dim[0] == A.dim[1]);
    assert(R.dim[1] == A.dim[1]);
    for(int j=0; j<A.dim[1]; j++) {
      Hierarchical Qj(A.dim[0], 1);
      for(int i = 0; i < A.dim[0]; i++) {
        Qj(i, 0) = Q(i, j);
      }
      Hierarchical Rjj(1, 1);
      Rjj(0, 0) = R(j, j);
      A.col_qr(j, Qj, Rjj);
      R(j, j) = Rjj(0, 0);
      for(int i=0; i<A.dim[0]; i++) {
        Q(i, j) = Qj(i, 0);
      }
      Hierarchical TrQj(Qj);
      transpose(TrQj);
      for(int k=j+1; k<A.dim[1]; k++) {
        //Take k-th column
        Hierarchical Ak(A.dim[0], 1);
        for(int i=0; i<A.dim[0]; i++) {
          Ak(i, 0) = A(i, k);
        }
        Hierarchical Rjk(1, 1);
        Rjk(0, 0) = R(j, k);
        gemm(TrQj, Ak, Rjk, 1, 1); //Rjk = Q*j^T x A*k
        R(j, k) = Rjk(0, 0);
        gemm(Qj, Rjk, Ak, -1, 1); //A*k = A*k - Q*j x Rjk
        for(int i=0; i<A.dim[0]; i++) {
          A(i, k) = Ak(i, 0);
        }
      }
    }
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(qr_omm, void, Node& A, Node& Q, Node& R) {
    std::cerr << "qr(";
    std::cerr << A.type() << "," << Q.type() << "," << R.type();
    std::cerr << ") undefined." << std::endl;
    abort();
  } END_SPECIALIZATION;


  BEGIN_SPECIALIZATION(need_split_omm, bool, const Hierarchical& A) {
    return true;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(need_split_omm, bool, const Node& A) {
    return false;
  } END_SPECIALIZATION;


  BEGIN_SPECIALIZATION(make_left_orthogonal_omm, dense_tuple, const Dense& A) {
    std::vector<double> x;
    Dense Id(identity, x, A.dim[0], A.dim[0]);
    Dense _A(A);
    return {Id, _A};\
  } END_SPECIALIZATION

  BEGIN_SPECIALIZATION(make_left_orthogonal_omm, dense_tuple, const LowRank& A) {
    Dense Au(A.U);
    Dense Qu(A.U.dim[0], A.U.dim[1]);
    Dense Ru(A.U.dim[1], A.U.dim[1]);
    qr(Au, Qu, Ru);
    Dense RS(Ru.dim[0], A.S.dim[1]);
    gemm(Ru, A.S, RS, 1, 1);
    Dense RSV(RS.dim[0], A.V.dim[1]);
    gemm(RS, A.V, RSV, 1, 1);
    return {Qu, RSV};
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_left_orthogonal_omm, dense_tuple, const Node& A) {
    std::cerr << "make_left_orthogonal(";
    std::cerr << A.type();
    std::cerr << ") undefined." << std::endl;
    abort();
  } END_SPECIALIZATION;


  BEGIN_SPECIALIZATION(update_splitted_size_omm, void, const Hierarchical& A, int& rows, int& cols) {
    rows += A.dim[0];
    cols = A.dim[1];
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(update_splitted_size_omm, void, const Node& A, int& rows, int& cols) {
    rows++;
  } END_SPECIALIZATION;


  BEGIN_SPECIALIZATION(split_by_column_omm, NodeProxy, const Dense& A, Hierarchical& storage, int& currentRow) {
    Hierarchical splitted(A, 1, storage.dim[1]);
    for(int i=0; i<storage.dim[1]; i++)
      storage(currentRow, i) = splitted(0, i);
    currentRow++;
    return Dense(0, 0);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(split_by_column_omm, NodeProxy, const LowRank& A, Hierarchical& storage, int& currentRow) {
    LowRank _A(A);
    Dense Qu(_A.U.dim[0], _A.U.dim[1]);
    Dense Ru(_A.U.dim[1], _A.U.dim[1]);
    qr(_A.U, Qu, Ru);
    Dense RS(Ru.dim[0], _A.S.dim[1]);
    gemm(Ru, _A.S, RS, 1, 0);
    Dense RSV(RS.dim[0], _A.V.dim[1]);
    gemm(RS, _A.V, RSV, 1, 0);
    //Split R*S*V
    Hierarchical splitted(RSV, 1, storage.dim[1]);
    for(int i=0; i<storage.dim[1]; i++) {
      storage(currentRow, i) = splitted(0, i);
    }
    currentRow++;
    return Qu;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(split_by_column_omm, NodeProxy, const Hierarchical& A, Hierarchical& storage, int& currentRow) {
    for(int i=0; i<A.dim[0]; i++) {
      for(int j=0; j<A.dim[1]; j++) {
        storage(currentRow, j) = A(i, j);
      }
      currentRow++;
    }
    return Dense(0, 0);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(split_by_column_omm, NodeProxy, const Node& A, Node& storage, int& currentRow) {
    std::cerr << "split_by_column(";
    std::cerr << A.type() << "," << storage.type() << ",int";
    std::cerr << ") undefined." << std::endl;
    abort();
  } END_SPECIALIZATION;


  BEGIN_SPECIALIZATION(concat_columns_omm, NodeProxy, const Dense& A, const Hierarchical& splitted, int& currentRow, const Dense& Q) {
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
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(concat_columns_omm, NodeProxy, const LowRank& A, const Hierarchical& splitted, int& currentRow, const Dense& Q) {
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
    _A.U = Q;
    _A.V = concatenatedRow;
    _A.S = Dense(identity, x, _A.rank, _A.rank);
    currentRow++;
    return _A;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(concat_columns_omm, NodeProxy, const Hierarchical& A, const Hierarchical& splitted, int& currentRow, const Dense& Q) {
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
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(concat_columns_omm, NodeProxy, const Node& A, const Node& splitted, int& currentRow, const Node& Q) {
    std::cerr << "concat_columns(";
    std::cerr << A.type() << "," << splitted.type() << ",int," << Q.type();
    std::cerr << ") undefined." << std::endl;
    abort();
  } END_SPECIALIZATION;


  BEGIN_SPECIALIZATION(zero_lowtri_omm, void, Dense& A) {
    for(int i=0; i<A.dim[0]; i++)
      for(int j=0; j<i; j++)
        A(i,j) = 0.0;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(zero_lowtri_omm, void, Node& A) {
    std::cerr << "zero_lowtri_omm(";
    std::cerr << A.type();
    std::cerr << ") undefined." << std::endl;
    abort();
  } END_SPECIALIZATION;


  BEGIN_SPECIALIZATION(zero_whole_omm, void, Dense& A) {
    A = 0.0;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(zero_whole_omm, void, LowRank& A) {
    A.U = 0.0; for(int i=0; i<std::min(A.U.dim[0], A.U.dim[1]); i++) A.U(i, i) = 1.0;
    A.S = 0.0;
    A.V = 0.0; for(int i=0; i<std::min(A.V.dim[0], A.V.dim[1]); i++) A.V(i, i) = 1.0;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(zero_whole_omm, void, Node& A) {
    std::cerr << "zero_whole_omm(";
    std::cerr << A.type();
    std::cerr << ") undefined." << std::endl;
    abort();
  } END_SPECIALIZATION;

} // namespace hicma
