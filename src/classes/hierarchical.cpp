#include "hicma/classes/hierarchical.h"

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/functions.h"
#include "hicma/operations/gemm.h"
#include "hicma/operations/get_dim.h"
#include "hicma/operations/qr.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/util/print.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>

#include "yorel/multi_methods.hpp"

namespace hicma {

  Hierarchical::Hierarchical() {
    MM_INIT();
    dim[0]=0; dim[1]=0;
  }

  Hierarchical::Hierarchical(
    const int ni_level,
    const int nj_level,
    const int i_abs,
    const int j_abs,
    const int level
  ) : Node(i_abs, j_abs, level) {
    MM_INIT();
    dim[0]=ni_level; dim[1]=nj_level; data.resize(dim[0]*dim[1]);
  }

  Hierarchical::Hierarchical(const Dense& A, const int m, const int n) : Node(A.i_abs,A.j_abs,A.level) {
    MM_INIT();
    dim[0]=m; dim[1]=n;
    data.resize(dim[0]*dim[1]);
    int ni = A.dim[0];
    int nj = A.dim[1];
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        int ni_child = ni/dim[0];
        if ( i == dim[0]-1 ) ni_child = ni - (ni/dim[0]) * (dim[0]-1);
        int nj_child = nj/dim[1];
        if ( j == dim[1]-1 ) nj_child = nj - (nj/dim[1]) * (dim[1]-1);
        int i_begin = ni/dim[0] * i;
        int j_begin = nj/dim[1] * j;
        int i_abs_child = A.i_abs * dim[0] + i;
        int j_abs_child = A.j_abs * dim[1] + j;
        Dense D(ni_child, nj_child, A.level+1, i_abs_child, j_abs_child);
        for (int ic=0; ic<ni_child; ic++) {
          for (int jc=0; jc<nj_child; jc++) {
            D(ic,jc) = A(ic+i_begin,jc+j_begin);
          }
        }
        (*this)(i,j) = std::move(D);
      }
    }
  }

  Hierarchical::Hierarchical(const LowRank& A, const int m, const int n) : Node(A.i_abs,A.j_abs,A.level) {
    MM_INIT();
    dim[0]=m; dim[1]=n;
    data.resize(dim[0]*dim[1]);
    int ni = A.dim[0];
    int nj = A.dim[1];
    int rank = A.rank;
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        int ni_child = ni/dim[0];
        if ( i == dim[0]-1 ) ni_child = ni - (ni/dim[0]) * (dim[0]-1);
        int nj_child = nj/dim[1];
        if ( j == dim[1]-1 ) nj_child = nj - (nj/dim[1]) * (dim[1]-1);
        int i_begin = ni/dim[0] * i;
        int j_begin = nj/dim[1] * j;
        int i_abs_child = A.i_abs * dim[0] + i;
        int j_abs_child = A.j_abs * dim[1] + j;
        LowRank LR(ni_child, nj_child, rank, A.level+1, i_abs_child, j_abs_child);
        for (int ic=0; ic<ni_child; ic++) {
          for (int kc=0; kc<rank; kc++) {
            LR.U(ic,kc) = A.U(ic+i_begin,kc);
          }
        }
        LR.S = A.S;
        for (int kc=0; kc<rank; kc++) {
          for (int jc=0; jc<nj_child; jc++) {
            LR.V(kc,jc) = A.V(kc,jc+j_begin);
          }
        }
        (*this)(i,j) = std::move(LR);
      }
    }
  }

  Hierarchical::Hierarchical(
                             void (*func)(
                                          std::vector<double>& data,
                                          std::vector<double>& x,
                                          const int& ni,
                                          const int& nj,
                                          const int& i_begin,
                                          const int& j_begin
                                          ),
                             std::vector<double>& x,
                             const int ni,
                             const int nj,
                             const int rank,
                             const int nleaf,
                             const int admis,
                             const int ni_level,
                             const int nj_level,
                             const int i_begin,
                             const int j_begin,
                             const int _i_abs,
                             const int _j_abs,
                             const int _level
                             ) : Node(_i_abs,_j_abs,_level) {
    MM_INIT();
    if ( !level ) {
      assert(int(x.size()) == std::max(ni,nj));
      std::sort(x.begin(),x.end());
    }
    dim[0] = std::min(ni_level,ni);
    dim[1] = std::min(nj_level,nj);
    data.resize(dim[0]*dim[1]);
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        int ni_child = ni/dim[0];
        if ( i == dim[0]-1 ) ni_child = ni - (ni/dim[0]) * (dim[0]-1);
        int nj_child = nj/dim[1];
        if ( j == dim[1]-1 ) nj_child = nj - (nj/dim[1]) * (dim[1]-1);
        int i_begin_child = i_begin + ni/dim[0] * i;
        int j_begin_child = j_begin + nj/dim[1] * j;
        int i_abs_child = i_abs * dim[0] + i;
        int j_abs_child = j_abs * dim[1] + j;
        if (
            std::abs(i_abs_child - j_abs_child) <= admis // Check regular admissibility
            || (nj == 1 || ni == 1) ) { // Check if vector, and if so do not use LowRank
          if ( ni_child/ni_level < nleaf && nj_child/nj_level < nleaf ) {
            (*this)(i,j) = Dense(
                                 func,
                                 x,
                                 ni_child,
                                 nj_child,
                                 i_begin_child,
                                 j_begin_child,
                                 i_abs_child,
                                 j_abs_child,
                                 level+1
                                 );
          }
          else {
            (*this)(i,j) = Hierarchical(
                                        func,
                                        x,
                                        ni_child,
                                        nj_child,
                                        rank,
                                        nleaf,
                                        admis,
                                        ni_level,
                                        nj_level,
                                        i_begin_child,
                                        j_begin_child,
                                        i_abs_child,
                                        j_abs_child,
                                        level+1
                                        );
          }
        }
        else {
          Dense A = Dense(
                          func,
                          x,
                          ni_child,
                          nj_child,
                          i_begin_child,
                          j_begin_child,
                          i_abs_child,
                          j_abs_child,
                          level+1
                          );
          rsvd_push((*this)(i,j), A, rank);
        }
      }
    }
  }

  Hierarchical::Hierarchical(const Hierarchical& A)
    : Node(A.i_abs,A.j_abs,A.level), data(A.data) {
    MM_INIT();
    dim[0]=A.dim[0]; dim[1]=A.dim[1];
  }

  Hierarchical::Hierarchical(Hierarchical&& A) {
    MM_INIT();
    swap(*this, A);
  }

  Hierarchical* Hierarchical::clone() const {
    return new Hierarchical(*this);
  }

  Hierarchical* Hierarchical::move_clone() {
    return new Hierarchical(std::move(*this));
  }

  void swap(Hierarchical& A, Hierarchical& B) {
    using std::swap;
    swap(A.data, B.data);
    swap(A.dim, B.dim);
    swap(A.i_abs, B.i_abs);
    swap(A.j_abs, B.j_abs);
    swap(A.level, B.level);
  }

  const NodeProxy& Hierarchical::operator[](const int i) const {
    assert(i < dim[0]*dim[1]);
    return data[i];
  }

  NodeProxy& Hierarchical::operator[](const int i) {
    assert(i < dim[0]*dim[1]);
    return data[i];
  }

  const NodeProxy& Hierarchical::operator()(const int i, const int j) const {
    assert(i < dim[0]);
    assert(j < dim[1]);
    return data[i*dim[1]+j];
  }

  NodeProxy& Hierarchical::operator()(const int i, const int j) {
    assert(i < dim[0]);
    assert(j < dim[1]);
    return data[i*dim[1]+j];
  }

  const char* Hierarchical::type() const { return "Hierarchical"; }

  void Hierarchical::blr_col_qr(Hierarchical& Q, Hierarchical& R) {
    assert(dim[1] == 1);
    assert(Q.dim[0] == dim[0]);
    assert(Q.dim[1] == 1);
    assert(R.dim[0] == 1);
    assert(R.dim[1] == 1);
    Hierarchical Qu(dim[0], 1);
    Hierarchical B(dim[0], 1);
    for(int i=0; i<dim[0]; i++) {
      std::tie(Qu(i, 0), B(i, 0)) = make_left_orthogonal((*this)(i, 0));
    }
    Dense DB(B);
    Dense Qb(DB.dim[0], DB.dim[1]);
    Dense Rb(DB.dim[1], DB.dim[1]);
    qr(DB, Qb, Rb);
    R(0, 0) = std::move(Rb);
    //Slice Qb based on B
    Hierarchical HQb(B.dim[0], B.dim[1]);
    int rowOffset = 0;
    for(int i=0; i<HQb.dim[0]; i++) {
      int nrows = get_n_rows(B(i, 0));
      int ncols = get_n_cols(B(i, 0));
      Dense Qbi(nrows, ncols);
      for(int row=0; row<nrows; row++) {
        for(int col=0; col<ncols; col++) {
          Qbi(row, col) = Qb(rowOffset + row, col);
        }
      }
      // Using move should now make a difference. Why is this not auto-optimized?
      HQb(i, 0) = std::move(Qbi);
      rowOffset += nrows;
    }
    for(int i=0; i<dim[0]; i++) {
      gemm(Qu(i, 0), HQb(i, 0), Q(i, 0), 1, 0);
    }
  }

  void Hierarchical::split_col(Hierarchical& QL) {
    assert(dim[1] == 1);
    assert(QL.dim[0] == dim[0]);
    assert(QL.dim[1] == 1);
    int rows = 0;
    int cols = 1;
    for(int i=0; i<dim[0]; i++) {
      update_splitted_size((*this)(i, 0), rows, cols);
    }
    Hierarchical spA(rows, cols);
    int curRow = 0;
    for(int i=0; i<dim[0]; i++) {
      QL(i, 0) = split_by_column((*this)(i, 0), spA, curRow);
    }
    swap(*this, spA);
  }

  void Hierarchical::restore_col(const Hierarchical& Sp, const Hierarchical& QL) {
    assert(dim[1] == 1);
    assert(dim[0] == QL.dim[0]);
    assert(QL.dim[1] == 1);
    Hierarchical restoredA(dim[0], dim[1]);
    int curSpRow = 0;
    for(int i=0; i<dim[0]; i++) {
      restoredA(i, 0) = concat_columns((*this)(i, 0), Sp, curSpRow, QL(i, 0));
    }
    swap(*this, restoredA);
  }

  void Hierarchical::col_qr(const int j, Hierarchical& Q, Hierarchical &R) {
    assert(Q.dim[0] == dim[0]);
    assert(Q.dim[1] == 1);
    assert(R.dim[0] == 1);
    assert(R.dim[1] == 1);
    bool split = false;
    Hierarchical Aj(dim[0], 1);
    for(int i=0; i<dim[0]; i++) {
      Aj(i, 0) = (*this)(i, j);
      split |= need_split(Aj(i, 0));
    }
    if(!split) {
      Aj.blr_col_qr(Q, R);
    }
    else {
      Hierarchical QL(dim[0], 1); //Stored Q for splitted lowrank blocks
      Aj.split_col(QL);
      Hierarchical SpQj(Aj);
      qr(Aj, SpQj, R(0, 0));
      Q.restore_col(SpQj, QL);
    }
  }

} // namespace hicma
