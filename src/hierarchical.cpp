#include "hicma/node_proxy.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/functions.h"
#include "hicma/operations.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/util/print.h"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "yorel/multi_methods.hpp"

namespace hicma {

  Hierarchical::Hierarchical() {
    MM_INIT();
    dim[0]=0; dim[1]=0;
  }

  Hierarchical::Hierarchical(const int m) {
    MM_INIT();
    dim[0]=m; dim[1]=1; data.resize(dim[0]);
  }

  Hierarchical::Hierarchical(const int m, const int n) {
    MM_INIT();
    dim[0]=m; dim[1]=n; data.resize(dim[0]*dim[1]);
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

  void swap(Hierarchical& A, Hierarchical& B) {
    using std::swap;
    swap(A.data, B.data);
    swap(A.dim, B.dim);
    swap(A.i_abs, B.i_abs);
    swap(A.j_abs, B.j_abs);
    swap(A.level, B.level);
  }

  const NodeProxy& Hierarchical::operator[](const int i) const {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  NodeProxy& Hierarchical::operator[](const int i) {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  const NodeProxy& Hierarchical::operator()(const int i, const int j) const {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  NodeProxy& Hierarchical::operator()(const int i, const int j) {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  const char* Hierarchical::type() const { return "Hierarchical"; }

  double Hierarchical::norm() const {
    double l2 = 0;
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        l2 += (*this)(i,j).norm();
      }
    }
    return l2;
  }

  void Hierarchical::print() const {
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        std::cout << (*this)(i, j).type() << " (" << i << "," << j << ")" << std::endl;
        (*this)(i,j).print();
      }
      std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
  }

  void Hierarchical::transpose() {
    Hierarchical C(dim[1], dim[0]);
    for(int i=0; i<dim[0]; i++) {
      for(int j=0; j<dim[1]; j++) {
        swap((*this)(i, j), C(j, i));
        C(j, i).transpose();
      }
    }
    swap(*this, C);
  }

  // void Hierarchical::blr_col_qr(Hierarchical& Q, Hierarchical& R) {
  //   assert(dim[1] == 1 && Q.dim[0] == dim[0] && Q.dim[1] == 1 && R.dim[0] == 1 && R.dim[1] == 1);
  //   Hierarchical Qu(dim[0], 1);
  //   Hierarchical B(dim[0], 1);
  //   for(int i=0; i<dim[0]; i++) {
  //     assert((*this)(i, 0).is(HICMA_DENSE) || (*this)(i, 0).is(HICMA_LOWRANK));
  //     if((*this)(i, 0).is(HICMA_DENSE)) {
  //       Dense Ai(static_cast<Dense&>(*(*this)(i, 0).ptr));
  //       std::vector<double> x;
  //       Qu(i, 0) = Dense(identity, x, Ai.dim[0], Ai.dim[0]);
  //       B(i, 0) = Ai;
  //     }
  //     else if((*this)(i, 0).is(HICMA_LOWRANK)) {
  //       LowRank Ai(static_cast<LowRank&>(*(*this)(i, 0).ptr));
  //       Dense Qui(Ai.U.dim[0], Ai.U.dim[1]);
  //       Dense Rui(Ai.U.dim[1], Ai.U.dim[1]);
  //       Ai.U.qr(Qui, Rui);
  //       Qu(i, 0) = Qui;
  //       Dense RS(Rui.dim[0], Ai.S.dim[1]);
  //       gemm(Rui, Ai.S, RS, 1, 1);
  //       Dense RSV(RS.dim[0], Ai.V.dim[1]);
  //       gemm(RS, Ai.V, RSV, 1, 1);
  //       B(i, 0) = RSV;
  //     }
  //   }
  //   Dense DB(B);
  //   Dense Qb(DB.dim[0], DB.dim[1]);
  //   Dense Rb(DB.dim[1], DB.dim[1]);
  //   DB.qr(Qb, Rb);
  //   R(0, 0) = Rb;
  //   //Slice Qb based on B
  //   Hierarchical HQb(B.dim[0], B.dim[1]);
  //   int rowOffset = 0;
  //   for(int i=0; i<HQb.dim[0]; i++) {
  //     Dense Bi(B(i, 0));
  //     Dense Qbi(Bi.dim[0], Bi.dim[1]);
  //     for(int row=0; row<Bi.dim[0]; row++) {
  //       for(int col=0; col<Bi.dim[1]; col++) {
  //         Qbi(row, col) = Qb(rowOffset + row, col);
  //       }
  //     }
  //     HQb(i, 0) = Qbi;
  //     rowOffset += Bi.dim[0];
  //   }
  //   for(int i=0; i<dim[0]; i++) {
  //     gemm(Qu(i, 0), HQb(i, 0), Q(i, 0), 1, 0);
  //   }
  // }

  // void Hierarchical::split_col(Hierarchical& QL) {
  //   assert(dim[1] == 1 && QL.dim[0] == dim[0] && QL.dim[1] == 1);
  //   int rows = 0;
  //   int cols = 1;
  //   for(int i=0; i<dim[0]; i++) {
  //     QL(i, 0) = Dense(0, 0);
  //     if((*this)(i, 0).is(HICMA_HIERARCHICAL)) {
  //       Hierarchical Ai(static_cast<Hierarchical&>(*(*this)(i, 0).ptr));
  //       rows += Ai.dim[0];
  //       cols = Ai.dim[1];
  //     }
  //     else {
  //       rows++;
  //     }
  //   }
  //   Hierarchical spA(rows, cols);
  //   int curRow = 0;
  //   for(int i=0; i<dim[0]; i++) {
  //     if((*this)(i, 0).is(HICMA_HIERARCHICAL)) {
  //       Hierarchical Ai(static_cast<Hierarchical&>(*(*this)(i, 0).ptr));
  //       for(int j=0; j<Ai.dim[0]; j++) {
  //         for(int k=0; k<Ai.dim[1]; k++) {
  //           spA(curRow, k) = Ai(j, k);
  //         }
  //         curRow++;
  //       }
  //     }
  //     else if((*this)(i, 0).is(HICMA_DENSE)) {
  //       Dense Ai((*this)(i, 0));
  //       Hierarchical BlockAi(Ai, 1, cols);
  //       for(int j=0; j<cols; j++) {
  //         spA(curRow, j) = BlockAi(0, j);
  //       }
  //       curRow++;
  //     }
  //     else if((*this)(i, 0).is(HICMA_LOWRANK)) {
  //       LowRank Ai(static_cast<LowRank&>(*(*this)(i, 0).ptr));
  //       Dense Qu(Ai.U.dim[0], Ai.U.dim[1]);
  //       Dense Ru(Ai.U.dim[1], Ai.U.dim[1]);
  //       Ai.U.qr(Qu, Ru);
  //       QL(i, 0) = Qu; //Store Q
  //       Dense RS(Ru.dim[0], Ai.S.dim[1]);
  //       gemm(Ru, Ai.S, RS, 1, 0);
  //       Dense RSV(RS.dim[0], Ai.V.dim[1]);
  //       gemm(RS, Ai.V, RSV, 1, 0);
  //       //Split R*S*V
  //       Hierarchical BlockRSV(RSV, 1, cols);
  //       for(int j=0; j<cols; j++) {
  //         spA(curRow, j) = BlockRSV(0, j);
  //       }
  //       curRow++;
  //     }
  //   }
  //   swap(*this, spA);
  // }

  // void Hierarchical::restore_col(const Hierarchical& Sp, const Hierarchical& QL) {
  //   assert(dim[1] == 1 && dim[0] == QL.dim[0] && QL.dim[1] == 1);
  //   int curSpRow = 0;
  //   for(int i=0; i<dim[0]; i++) {
  //     if((*this)(i, 0).is(HICMA_HIERARCHICAL)) {
  //       //In case of hierarchical, just put element in respective cells
  //       Hierarchical Ai(static_cast<Hierarchical&>(*(*this)(i, 0).ptr));
  //       assert(Sp.dim[1] == Ai.dim[1]);
  //       for(int p=0; p<Ai.dim[0]; p++) {
  //         for(int q=0; q<Ai.dim[1]; q++) {
  //           Ai(p, q) = Sp(curSpRow, q);
  //         }
  //         curSpRow++;
  //       }
  //       (*this)(i, 0) = Ai;
  //     }
  //     else if((*this)(i, 0).is(HICMA_DENSE)) {
  //       //In case of dense, combine the spllited dense matrices into one dense matrix
  //       Dense Ai(static_cast<Dense&>(*(*this)(i, 0).ptr));
  //       Hierarchical SpCurRow(1, Sp.dim[1]);
  //       for(int q=0; q<Sp.dim[1]; q++) {
  //         SpCurRow(0, q) = Sp(curSpRow, q);
  //       }
  //       Dense SpCurRowCombined(SpCurRow);
  //       assert(Ai.dim[0] == SpCurRowCombined.dim[0] && Ai.dim[1] == SpCurRowCombined.dim[1]);
  //       (*this)(i, 0) = SpCurRowCombined;
  //       curSpRow++;
  //     }
  //     else if((*this)(i, 0).is(HICMA_LOWRANK)) {
  //       //In case of lowrank, combine splitted dense matrices into single dense matrix
  //       //Then form a lowrank matrix with the stored QL
  //       LowRank Ai(static_cast<LowRank&>(*(*this)(i, 0).ptr));
  //       Hierarchical SpCurRow(1, Sp.dim[1]);
  //       for(int q=0; q<Sp.dim[1]; q++) {
  //         SpCurRow(0, q) = Sp(curSpRow, q);
  //       }
  //       Dense SpCurRowCombined(SpCurRow);
  //       Dense QLi(QL(i, 0));
  //       assert(QLi.dim[0] == Ai.dim[0] && QLi.dim[1] == Ai.rank && SpCurRowCombined.dim[0] == Ai.rank && SpCurRowCombined.dim[1] == Ai.dim[1]);
  //       Ai.U = QLi;
  //       Ai.V = SpCurRowCombined;
  //       //Fill S with identity matrix
  //       std::vector<double> x;
  //       Ai.S = Dense(identity, x, Ai.rank, Ai.rank);
  //       (*this)(i, 0) = Ai;
  //       curSpRow++;
  //     }
  //   }
  // }

  // void Hierarchical::col_qr(const int j, Hierarchical& Q, Hierarchical &R) {
  //   assert(Q.dim[0] == dim[0] && Q.dim[1] == 1 && R.dim[0] == 1 && R.dim[1] == 1);
  //   bool split = false;
  //   Hierarchical Aj(dim[0], 1);
  //   for(int i=0; i<dim[0]; i++) {
  //     Aj(i, 0) = (*this)(i, j);
  //     if((*this)(i, j).is(HICMA_HIERARCHICAL)) {
  //       split = true;
  //     }
  //   }
  //   if(!split) {
  //     Aj.blr_col_qr(Q, R);
  //   }
  //   else {
  //     Hierarchical QL(dim[0], 1); //Stored Q for splitted lowrank blocks
  //     Hierarchical Rjj(static_cast<Hierarchical&>(*R(0, 0).ptr));
  //     Aj.split_col(QL);
  //     Hierarchical SpQj(Aj);
  //     Aj.qr(SpQj, Rjj);
  //     Q.restore_col(SpQj, QL);
  //     R(0, 0) = Rjj;
  //   }
  // }

  // void Hierarchical::qr(Hierarchical& Q, Hierarchical &R) {
  //   assert(Q.dim[0] == dim[0] && Q.dim[1] == dim[1] && R.dim[0] == dim[1] && R.dim[1] == dim[1]);
  //   for(int j=0; j<dim[1]; j++) {
  //     Hierarchical Qj(dim[0], 1);
  //     for(int i = 0; i < dim[0]; i++) {
  //       Qj(i, 0) = Q(i, j);
  //     }
  //     Hierarchical Rjj(1, 1);
  //     Rjj(0, 0) = R(j, j);
  //     col_qr(j, Qj, Rjj);
  //     R(j, j) = Rjj(0, 0);
  //     for(int i=0; i<dim[0]; i++) {
  //       Q(i, j) = Qj(i, 0);
  //     }
  //     Hierarchical TrQj(Qj);
  //     TrQj.transpose();
  //     for(int k=j+1; k<dim[1]; k++) {
  //       //Take k-th column
  //       Hierarchical Ak(dim[0], 1);
  //       for(int i=0; i<dim[0]; i++) {
  //         Ak(i, 0) = (*this)(i, k);
  //       }
  //       Hierarchical Rjk(1, 1);
  //       Rjk(0, 0) = R(j, k);
  //       gemm(TrQj, Ak, Rjk, 1, 1); //Rjk = Q*j^T x A*k
  //       R(j, k) = Rjk(0, 0);
  //       gemm(Qj, Rjk, Ak, -1, 1); //A*k = A*k - Q*j x Rjk
  //       for(int i=0; i<dim[0]; i++) {
  //         (*this)(i, k) = Ak(i, 0);
  //       }
  //     }
  //   }
  // }

  // void Hierarchical::geqrt(Hierarchical& T) {
  //   for(int k = 0; k < dim[1]; k++) {
  //     (*this)(k, k).geqrt(T(k, k));
  //     for(int j = k+1; j < dim[1]; j++) {
  //       (*this)(k, j).larfb((*this)(k, k), T(k, k), true);
  //     }
  //     int dim0 = -1;
  //     int dim1 = -1;
  //     if((*this)(k, k).is(HICMA_HIERARCHICAL)) {
  //       Hierarchical Akk(static_cast<Hierarchical&>(*(*this)(k, k).ptr));
  //       dim0 = Akk.dim[0];
  //       dim1 = Akk.dim[1];
  //     }
  //     for(int i = k+1; i < dim[0]; i++) {
  //       if((*this)(k, k).is(HICMA_HIERARCHICAL)) {
  //         if((*this)(i, k).is(HICMA_DENSE))
  //           (*this)(i, k) = Hierarchical(static_cast<Dense&>(*(*this)(i, k).ptr), dim0, dim1);
  //         if(T(i, k).is(HICMA_DENSE))
  //           T(i, k) = Hierarchical(static_cast<Dense&>(*T(i, k).ptr), dim0, dim1);
  //       }
  //       tpqrt((*this)(k, k), (*this)(i, k), T(i, k));
  //       for(int j = k+1; j < dim[1]; j++) {
  //         tpmqrt((*this)(i, k), T(i, k), (*this)(k, j), (*this)(i, j), true);
  //       }
  //     }
  //   }
  // }

  // void Hierarchical::larfb(const Dense& Y, const Dense& T, const bool trans) {
  //   Dense _Y(Y);
  //   for(int i = 0; i < _Y.dim[0]; i++) {
  //     for(int j = i; j < _Y.dim[1]; j++) {
  //       if(i == j) _Y(i, j) = 1.0;
  //       else _Y(i, j) = 0.0;
  //     }
  //   }
  //   Dense YT(_Y.dim[0], T.dim[1]);
  //   gemm(_Y, T, YT, CblasNoTrans, trans ? CblasTrans : CblasNoTrans, 1, 1);
  //   Dense YTYt(YT.dim[0], _Y.dim[0]);
  //   gemm(YT, _Y, YTYt, CblasNoTrans, CblasTrans, 1, 1);
  //   Hierarchical C(*this);
  //   gemm(YTYt, C, *this, -1, 1);
  // }

  // void Hierarchical::larfb(const Hierarchical& Y, const Hierarchical& T, const bool trans) {
  //   if(trans) {
  //     for(int k = 0; k < dim[1]; k++) {
  //       for(int j = k; j < dim[1]; j++) {
  //         (*this)(k, j).larfb(Y(k, k), T(k, k), trans);
  //       }
  //       for(int i = k+1; i < dim[0]; i++) {
  //         for(int j = k; j < dim[1]; j++) {
  //           tpmqrt(Y(i, k), T(i, k), (*this)(k, j), (*this)(i, j), trans);
  //         }
  //       }
  //     }
  //   }
  //   else {
  //     for(int k = dim[1]-1; k >= 0; k--) {
  //       for(int i = dim[0]-1; i > k; i--) {
  //         for(int j = k; j < dim[1]; j++) {
  //           tpmqrt(Y(i, k), T(i, k), (*this)(k, j), (*this)(i, j), trans);
  //         }
  //       }
  //       for(int j = k; j < dim[1]; j++) {
  //         (*this)(k, j).larfb(Y(k, k), T(k, k), trans);
  //       }
  //     }
  //   }
  // }

}
