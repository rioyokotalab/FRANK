#include "hicma/operations/qr.h"

#include "hicma/node.h"
#include "hicma/node_proxy.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations/gemm.h"

#include <cassert>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{
  void qr(NodeProxy& A, NodeProxy& Q, NodeProxy& R) {
    qr(*A.ptr, *Q.ptr, *R.ptr);
  }
  void qr(NodeProxy& A, NodeProxy& Q, Node& R) {
    qr(*A.ptr, *Q.ptr, R);
  }
  void qr(NodeProxy& A, Node& Q, NodeProxy& R) {
    qr(*A.ptr, Q, *R.ptr);
  }
  void qr(NodeProxy& A, Node& Q, Node& R) {
    qr(*A.ptr, Q, R);
  }
  void qr(Node& A, NodeProxy& Q, NodeProxy& R) {
    qr(A, *Q.ptr, *R.ptr);
  }
  void qr(Node& A, NodeProxy& Q, Node& R) {
    qr(A, *Q.ptr, R);
  }
  void qr(Node& A, Node& Q, NodeProxy& R) {
    qr(A, Q, *R.ptr);
  }
  void qr(Node& A, Node& Q, Node& R) {
    qr_omm(A, Q, R);
  }

  bool need_split(const NodeProxy& A) {
    return need_split(*A.ptr.get());
  }
  bool need_split(const Node& A) {
    return need_split_omm(A);
  }

  void make_left_orthogonal(const NodeProxy& A, NodeProxy& L, NodeProxy& R) {
    make_left_orthogonal_omm(*A.ptr, L, R);
  }

  void update_splitted_size(const NodeProxy& A, int& rows, int& cols) {
    update_splitted_size(*A.ptr, rows, cols);
  }
  void update_splitted_size(const Node& A, int& rows, int& cols) {
    update_splitted_size_omm(A, rows, cols);
  }

  void split_by_column(const NodeProxy& A, Node& storage, int &currentRow, NodeProxy& Q) {
    split_by_column(*A.ptr, storage, currentRow, Q);
  }
  void split_by_column(const Node& A, Node& storage, int &currentRow, NodeProxy& Q) {
    split_by_column_omm(A, storage, currentRow, Q);
  }

  void concat_columns(const NodeProxy& A, const Node& splitted, NodeProxy& restored, int& currentRow, const NodeProxy& Q) {
    concat_columns(*A.ptr, splitted, restored, currentRow, *Q.ptr);
  }
  void concat_columns(const Node& A, const Node& splitted, NodeProxy& restored, int& currentRow, const Node& Q) {
    concat_columns_omm(A, splitted, restored, currentRow, Q);
  }

  BEGIN_SPECIALIZATION(qr_omm, void, Dense& A, Dense& Q, Dense& R) {
    std::vector<double> tau(A.dim[1]);
    for (int i=0; i<A.dim[1]; i++) Q[i*(A.dim[1])+i] = 1.0;
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.dim[0], A.dim[1], &A[0], A.dim[1], &tau[0]);
    LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'N', A.dim[0], A.dim[1], A.dim[1],
                   &A[0], A.dim[1], &tau[0], &Q[0], A.dim[1]);
    for(int i=0; i<A.dim[1]; i++) {
      for(int j=0; j<A.dim[1]; j++) {
        if(j>=i){
          R[i*(A.dim[1])+j] = A[i*(A.dim[1])+j];
        }
      }
    }
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
      TrQj.transpose();
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


  BEGIN_SPECIALIZATION(make_left_orthogonal_omm, void, const Dense& A, NodeProxy& L, NodeProxy& R) {
    std::vector<double> x;
    Dense Id(identity, x, A.dim[0], A.dim[0]);
    Dense _A(A);
    L = std::move(Id);
    R = std::move(_A);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_left_orthogonal_omm, void, const LowRank& A, NodeProxy& L, NodeProxy& R) {
    LowRank _A(A);
    Dense Qu(_A.U.dim[0], _A.U.dim[1]);
    Dense Ru(_A.U.dim[1], _A.U.dim[1]);
    qr(_A.U, Qu, Ru);
    L = std::move(Qu);
    Dense RS(Ru.dim[0], _A.S.dim[1]);
    gemm(Ru, _A.S, RS, 1, 1);
    Dense RSV(RS.dim[0], _A.V.dim[1]);
    gemm(RS, _A.V, RSV, 1, 1);
    R = std::move(RSV);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_left_orthogonal_omm, void, const Node& A, NodeProxy& L, NodeProxy& R) {
    std::cerr << "make_left_orthogonal(";
    std::cerr << A.type() << "," << L.type() << "," << R.type();
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


  BEGIN_SPECIALIZATION(split_by_column_omm, void, const Dense& A, Hierarchical& storage, int& currentRow, NodeProxy& Q) {
    Hierarchical splitted(A, 1, storage.dim[1]);
    for(int i=0; i<storage.dim[1]; i++)
      storage(currentRow, i) = splitted(0, i);
    currentRow++;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(split_by_column_omm, void, const LowRank& A, Hierarchical& storage, int& currentRow, NodeProxy& Q) {
    LowRank _A(A);
    Dense Qu(_A.U.dim[0], _A.U.dim[1]);
    Dense Ru(_A.U.dim[1], _A.U.dim[1]);
    qr(_A.U, Qu, Ru);
    Q = std::move(Qu); //Store Q
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
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(split_by_column_omm, void, const Hierarchical& A, Hierarchical& storage, int& currentRow, NodeProxy& Q) {
    for(int i=0; i<A.dim[0]; i++) {
      for(int j=0; j<A.dim[1]; j++) {
        storage(currentRow, j) = A(i, j);
      }
      currentRow++;
    }
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(split_by_column_omm, void, const Node& A, Node& storage, int& currentRow, NodeProxy& Q) {
    std::cerr << "split_by_column(";
    std::cerr << A.type() << "," << storage.type() << ",int," << Q.type();
    std::cerr << ") undefined." << std::endl;
    abort();
  } END_SPECIALIZATION;


  BEGIN_SPECIALIZATION(concat_columns_omm, void, const Dense& A, const Hierarchical& splitted, NodeProxy& restored, int& currentRow, const Dense& Q) {
    //In case of dense, combine the spllited dense matrices into one dense matrix
    Hierarchical SpCurRow(1, splitted.dim[1]);
    for(int i=0; i<splitted.dim[1]; i++) {
      SpCurRow(0, i) = splitted(currentRow, i);
    }
    Dense concatenatedRow(SpCurRow);
    assert(A.dim[0] == concatenatedRow.dim[0]);
    assert(A.dim[1] == concatenatedRow.dim[1]);
    restored = std::move(concatenatedRow);
    currentRow++;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(concat_columns_omm, void, const LowRank& A, const Hierarchical& splitted, NodeProxy& restored, int& currentRow, const Dense& Q) {
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
    LowRank _A(A);
    _A.U = Q;
    _A.V = concatenatedRow;
    _A.S = Dense(identity, x, _A.rank, _A.rank);
    restored = std::move(_A);
    currentRow++;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(concat_columns_omm, void, const Hierarchical& A, const Hierarchical& splitted, NodeProxy& restored, int& currentRow, const Dense& Q) {
    //In case of hierarchical, just put element in respective cells
    assert(splitted.dim[1] == A.dim[1]);
    Hierarchical concatenatedRow(A.dim[0], A.dim[1]);
    for(int i=0; i<A.dim[0]; i++) {
      for(int j=0; j<A.dim[1]; j++) {
        concatenatedRow(i, j) = splitted(currentRow, j);
      }
      currentRow++;
    }
    restored = std::move(concatenatedRow);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(concat_columns_omm, void, const Node& A, const Node& splitted, NodeProxy& restored, int& currentRow, const Node& Q) {
    std::cerr << "concat_columns(";
    std::cerr << A.type() << "," << splitted.type() << "," << restored.type() << ",int," << Q.type();
    std::cerr << ") undefined." << std::endl;
    abort();
  } END_SPECIALIZATION;

} // namespace hicma
