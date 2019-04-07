#include "any.h"
#include "low_rank.h"
#include "hierarchical.h"
#include "batch.h"
#include "print.h"
#include "functions.h"

#include <algorithm>
#include <cassert>
#include <iostream>

namespace hicma {

  Hierarchical::Hierarchical() {
    dim[0]=0; dim[1]=0;
  }

  Hierarchical::Hierarchical(const int m) {
    dim[0]=m; dim[1]=1; data.resize(dim[0]);
  }

  Hierarchical::Hierarchical(const int m, const int n) {
    dim[0]=m; dim[1]=n; data.resize(dim[0]*dim[1]);
  }

  Hierarchical::Hierarchical(const Dense& A, const int m, const int n) : Node(A.i_abs,A.j_abs,A.level) {
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
    dim[0]=A.dim[0]; dim[1]=A.dim[1];
  }

  Hierarchical::Hierarchical(Hierarchical&& A) {
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

  const Any& Hierarchical::operator[](const int i) const {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  Any& Hierarchical::operator[](const int i) {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  const Any& Hierarchical::operator()(const int i, const int j) const {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  Any& Hierarchical::operator()(const int i, const int j) {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  bool Hierarchical::is(const int enum_id) const {
    return enum_id == HICMA_HIERARCHICAL;
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
        if ((*this)(i,j).is(HICMA_DENSE)) {
          std::cout << "D (" << i << "," << j << ")" << std::endl;
        }
        else if ((*this)(i,j).is(HICMA_LOWRANK)) {
          std::cout << "L (" << i << "," << j << ")" << std::endl;
        }
        else if ((*this)(i,j).is(HICMA_HIERARCHICAL)) {
          std::cout << "H (" << i << "," << j << ")" << std::endl;
        }
        else {
          std::cout << "? (" << i << "," << j << ")" << std::endl;
        }
        (*this)(i,j).print();
      }
      std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
  }

  void Hierarchical::transpose() {
    Hierarchical Tr(dim[1], dim[0]);
    for(int i=0; i<dim[0]; i++) {
      for(int j=0; j<dim[1]; j++) {
        (*this)(i, j).transpose();
        Tr(j, i) = (*this)(i, j);
      }
    }
    swap(*this, Tr);
  }

  void Hierarchical::getrf() {
    for (int i=0; i<dim[0]; i++) {
      (*this)(i,i).getrf();
      for (int j=i+1; j<dim[0]; j++) {
        (*this)(i,j).trsm((*this)(i,i), 'l');
        (*this)(j,i).trsm((*this)(i,i), 'u');
      }
      for (int j=i+1; j<dim[0]; j++) {
        for (int k=i+1; k<dim[0]; k++) {
          (*this)(j,k).gemm((*this)(j,i), (*this)(i,k), -1, 1);
        }
      }
    }
  }

  void Hierarchical::trsm(const Dense& A, const char& uplo) {
    print_undefined(__func__, A.type(), this->type());
    abort();
  }

  void Hierarchical::trsm(const Hierarchical& A, const char& uplo) {
    if (dim[1] == 1) {
      switch (uplo) {
      case 'l' :
        for (int i=0; i<dim[0]; i++) {
          for (int j=0; j<i; j++) {
            (*this)[i].gemm(A(i,j), (*this)[j], -1, 1);
          }
          (*this)[i].trsm(A(i,i), 'l');
        }
        break;
      case 'u' :
        for (int i=dim[0]-1; i>=0; i--) {
          for (int j=dim[0]-1; j>i; j--) {
            (*this)[i].gemm(A(i,j), (*this)[j], -1, 1);
          }
          (*this)[i].trsm(A(i,i), 'u');
        }
        break;
      default :
        std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
        abort();
      }
    }
    else {
      switch (uplo) {
      case 'l' :
        for (int j=0; j<dim[1]; j++) {
          for (int i=0; i<dim[0]; i++) {
            (*this).gemm_row(A, *this, i, j, 0, i, -1, 1);
            (*this)(i,j).trsm(A(i,i), 'l');
          }
        }
        break;
      case 'u' :
        for (int i=0; i<dim[0]; i++) {
          for (int j=0; j<dim[1]; j++) {
            (*this).gemm_row(*this, A, i, j, 0, j, -1, 1);
            (*this)(i,j).trsm(A(j,j), 'u');
          }
        }
        break;
      default :
        std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
        abort();
      }
    }
  }

  void Hierarchical::gemm(const Dense& A, const Dense& B, const double& alpha, const double& beta) {
    print_undefined(__func__, A.type(), B.type(), this->type());
    abort();
  }

  void Hierarchical::gemm(const Dense& A, const LowRank& B, const double& alpha, const double& beta) {
    print_undefined(__func__, A.type(), B.type(), this->type());
    abort();
  }

  void Hierarchical::gemm(const Dense& A, const Hierarchical& B, const double& alpha, const double& beta) {
    const Hierarchical& AH = Hierarchical(A, dim[0], B.dim[0]);
    gemm(AH, B, alpha, beta);
  }

  void Hierarchical::gemm(const LowRank& A, const Dense& B, const double& alpha, const double& beta) {
    print_undefined(__func__, A.type(), B.type(), this->type());
    abort();
  }

  void Hierarchical::gemm(const LowRank& A, const LowRank& B, const double& alpha, const double& beta) {
    const Hierarchical& AH = Hierarchical(A, dim[0], dim[0]);
    const Hierarchical& BH = Hierarchical(B, dim[1], dim[1]);
    gemm(AH, BH, alpha, beta);
  }

  void Hierarchical::gemm(const LowRank& A, const Hierarchical& B, const double& alpha, const double& beta) {
    const Hierarchical& AH = Hierarchical(A, dim[0], B.dim[0]);
    gemm(AH, B, alpha, beta);
  }

  void Hierarchical::gemm(const Hierarchical& A, const Dense& B, const double& alpha, const double& beta) {
    const Hierarchical& BH = Hierarchical(B, A.dim[1], dim[1]);
    gemm(A, BH, alpha, beta);
  }

  void Hierarchical::gemm(const Hierarchical& A, const LowRank& B, const double& alpha, const double& beta) {
    const Hierarchical& BH = Hierarchical(B, A.dim[1], dim[1]);
    gemm(A, BH, alpha, beta);
  }

  void Hierarchical::gemm(const Hierarchical& A, const Hierarchical& B, const double& alpha, const double& beta) {
    assert(dim[0]==A.dim[0] && dim[1]==B.dim[1] && A.dim[1] == B.dim[0]);
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        (*this).gemm_row(A, B, i, j, 0, A.dim[1], alpha, beta);
      }
    }
  }

  void Hierarchical::gemm_row(
                              const Hierarchical& A, const Hierarchical& B,
                              const int& i, const int& j, const int& k_min, const int& k_max,
                              const double& alpha, const double& beta)
  {
    int rank = -1;
    if ((*this)(i,j).is(HICMA_LOWRANK)) {
      rank = static_cast<LowRank&>(*(*this)(i,j).ptr).rank;
      (*this)(i,j) = Dense(static_cast<LowRank&>(*(*this)(i,j).ptr));
    }
    for (int k=k_min; k<k_max; k++) {
      (*this)(i,j).gemm(A(i,k), B(k,j), alpha, beta);
    }
    if (rank != -1) {
      assert((*this)(i,j).is(HICMA_DENSE));
      (*this)(i,j) = LowRank(static_cast<Dense&>(*(*this)(i,j).ptr), rank);
    }
  }

  void Hierarchical::blr_col_qr(Hierarchical& Q, Dense& R) {
    assert(dim[1] == 1);
    Hierarchical Qu(dim[0], 1);
    Hierarchical B(dim[0], 1);
    for(int i=0; i<dim[0]; i++) {
      assert((*this)(i, 0).is(HICMA_DENSE) || (*this)(i, 0).is(HICMA_LOWRANK));
      if((*this)(i, 0).is(HICMA_DENSE)) {
        Dense Ai(static_cast<Dense&>(*(*this)(i, 0).ptr));
        std::vector<double> x;
        Qu(i, 0) = Dense(identity, x, Ai.dim[0], Ai.dim[0]);
        B(i, 0) = Ai;
      }
      else if((*this)(i, 0).is(HICMA_LOWRANK)) {
        LowRank Ai(static_cast<LowRank&>(*(*this)(i, 0).ptr));
        Dense Qui(Ai.U.dim[0], Ai.U.dim[1]);
        Dense Rui(Ai.U.dim[1], Ai.U.dim[1]);
        Ai.U.qr(Qui, Rui);
        Qu(i, 0) = Qui;
        Dense RS(Rui.dim[0], Ai.S.dim[1]);
        RS.gemm(Rui, Ai.S, 1, 1);
        Dense RSV(RS.dim[0], Ai.V.dim[1]);
        RSV.gemm(RS, Ai.V, 1, 1);
        B(i, 0) = RSV;
      }
    }
    Dense DB(B);
    Dense Qb(DB.dim[0], DB.dim[1]);
    Dense Rb(DB.dim[1], DB.dim[1]);
    DB.qr(Qb, Rb);
    R = Rb;
    //Slice Qb based on B
    Hierarchical HQb(B.dim[0], B.dim[1]);
    int rowOffset = 0;
    for(int i=0; i<HQb.dim[0]; i++) {
      Dense Bi(B(i, 0));
      Dense Qbi(Bi.dim[0], Bi.dim[1]);
      for(int row=0; row<Bi.dim[0]; row++) {
        for(int col=0; col<Bi.dim[1]; col++) {
          Qbi(row, col) = Qb(rowOffset + row, col);
        }
      }
      HQb(i, 0) = Qbi;
      rowOffset += Bi.dim[0];
    }
    Hierarchical Qj(dim[0], 1);
    for(int i=0; i<dim[0]; i++) {
      Qj(i, 0) = (*this)(i, 0);
      Qj(i, 0).gemm(Qu(i, 0), HQb(i, 0), 1, 0);
    }
    swap(Q, Qj);
  }

  void Hierarchical::split_col(Hierarchical& QL) {
    assert(dim[1] == 1);
    int rows = 0;
    int cols = 1;
    for(int i=0; i<dim[0]; i++) {
      QL(i, 0) = Dense(0, 0);
      if((*this)(i, 0).is(HICMA_HIERARCHICAL)) {
        Hierarchical Ai(static_cast<Hierarchical&>(*(*this)(i, 0).ptr));
        rows += Ai.dim[0];
        cols = Ai.dim[1];
      }
      else {
        rows++;
      }
    }
    Hierarchical spA(rows, cols);
    int curRow = 0;
    for(int i=0; i<dim[0]; i++) {
      if((*this)(i, 0).is(HICMA_HIERARCHICAL)) {
        Hierarchical Ai(static_cast<Hierarchical&>(*(*this)(i, 0).ptr));
        for(int j=0; j<Ai.dim[0]; j++) {
          for(int k=0; k<Ai.dim[1]; k++) {
            spA(curRow, k) = Ai(j, k);
          }
          curRow++;
        }
      }
      else if((*this)(i, 0).is(HICMA_DENSE)) {
        Dense Ai((*this)(i, 0));
        Hierarchical BlockAi(Ai, 1, cols);
        for(int j=0; j<cols; j++) {
          spA(curRow, j) = BlockAi(0, j);
        }
        curRow++;
      }
      else if((*this)(i, 0).is(HICMA_LOWRANK)) {
        LowRank Ai(static_cast<LowRank&>(*(*this)(i, 0).ptr));
        Dense Qu(Ai.U.dim[0], Ai.U.dim[1]);
        Dense Ru(Ai.U.dim[1], Ai.U.dim[1]);
        Ai.U.qr(Qu, Ru);
        QL(i, 0) = Qu; //Store Q
        Dense RS(Ru.dim[0], Ai.S.dim[1]);
        RS.gemm(Ru, Ai.S, 1, 0);
        Dense RSV(RS.dim[0], Ai.V.dim[1]);
        RSV.gemm(RS, Ai.V, 1, 0);
        //Split R*S*V
        Hierarchical BlockRSV(RSV, 1, cols);
        for(int j=0; j<cols; j++) {
          spA(curRow, j) = BlockRSV(0, j);
        }
        curRow++;
      }
    }
    swap(*this, spA);
  }

  void Hierarchical::restore_col(const Hierarchical& Sp, const Hierarchical& QL) {
    assert(dim[1] == 1 && dim[0] == QL.dim[0] && QL.dim[1] == 1);
    int curSpRow = 0;
    for(int i=0; i<dim[0]; i++) {
      if((*this)(i, 0).is(HICMA_HIERARCHICAL)) {
        //In case of hierarchical, just put element in respective cells
        Hierarchical Ai(static_cast<Hierarchical&>(*(*this)(i, 0).ptr));
        assert(Sp.dim[1] == Ai.dim[1]);
        for(int p=0; p<Ai.dim[0]; p++) {
          for(int q=0; q<Ai.dim[1]; q++) {
            Ai(p, q) = Sp(curSpRow, q);
          }
          curSpRow++;
        }
        (*this)(i, 0) = Ai;
      }
      else if((*this)(i, 0).is(HICMA_DENSE)) {
        //In case of dense, combine the spllited dense matrices into one dense matrix
        Dense Ai(static_cast<Dense&>(*(*this)(i, 0).ptr));
        Hierarchical SpCurRow(1, Sp.dim[1]);
        for(int q=0; q<Sp.dim[1]; q++) {
          SpCurRow(0, q) = Sp(curSpRow, q);
        }
        Dense SpCurRowCombined(SpCurRow);
        assert(Ai.dim[0] == SpCurRowCombined.dim[0] && Ai.dim[1] == SpCurRowCombined.dim[1]);
        (*this)(i, 0) = SpCurRowCombined;
        curSpRow++;
      }
      else if((*this)(i, 0).is(HICMA_LOWRANK)) {
        //In case of lowrank, combine splitted dense matrices into single dense matrix
        //Then form a lowrank matrix with the stored QL
        LowRank Ai(static_cast<LowRank&>(*(*this)(i, 0).ptr));
        Hierarchical SpCurRow(1, Sp.dim[1]);
        for(int q=0; q<Sp.dim[1]; q++) {
          SpCurRow(0, q) = Sp(curSpRow, q);
        }
        Dense SpCurRowCombined(SpCurRow);
        Dense QLi(QL(i, 0));
        assert(QLi.dim[0] == Ai.dim[0] && QLi.dim[1] == Ai.rank && SpCurRowCombined.dim[0] == Ai.rank && SpCurRowCombined.dim[1] == Ai.dim[1]);
        Ai.U = QLi;
        Ai.V = SpCurRowCombined;
        //Fill S with identity matrix
        std::vector<double> x;
        Ai.S = Dense(identity, x, Ai.rank, Ai.rank);
        (*this)(i, 0) = Ai;
        curSpRow++;
      }
    }
  }

  void Hierarchical::col_qr(const int j, Hierarchical& Q, Hierarchical &R) {
    assert(Q.dim[0] == dim[0] && Q.dim[1] == 1);
    bool split = false;
    Hierarchical Aj(dim[0], 1);
    for(int i=0; i<dim[0]; i++) {
      Aj(i, 0) = (*this)(i, j);
      if((*this)(i, j).is(HICMA_HIERARCHICAL)) {
        split = true;
      }
    }
    if(!split) {
      Dense DR;
      Aj.blr_col_qr(Q, DR);
      R(0, 0) = DR;
    }
    else {
      Hierarchical QL(dim[0], 1); //Stored Q for splitted lowrank blocks
      Hierarchical Rjj(static_cast<Hierarchical&>(*R(0, 0).ptr));
      Aj.split_col(QL);
      Hierarchical SpQj(Aj);
      Aj.qr(SpQj, Rjj);
      Q.restore_col(SpQj, QL);
      R(0, 0) = Rjj;
    }
  }

  void Hierarchical::qr(Hierarchical& Q, Hierarchical &R) {
    assert(Q.dim[0] == dim[0] && Q.dim[1] == dim[1] && R.dim[0] == dim[1] && R.dim[1] == dim[1]);
    for(int j=0; j<dim[1]; j++) {
      Hierarchical Qj(dim[0], 1);
      for(int i = 0; i < dim[0]; i++) {
        Qj(i, 0) = Q(i, j);
      }
      Hierarchical Rjj(1, 1);
      Rjj(0, 0) = R(j, j);
      (*this).col_qr(j, Qj, Rjj);
      R(j, j) = Rjj(0, 0);
      for(int i=0; i<dim[0]; i++) {
        Q(i, j) = Qj(i, 0);
      }
      Hierarchical TrQj(Qj);
      TrQj.transpose();
      for(int k=j+1; k<dim[1]; k++) {
        //Take k-th column
        Hierarchical Ak(dim[0], 1);
        for(int i=0; i<dim[0]; i++) {
          Ak(i, 0) = (*this)(i, k);
        }
        Hierarchical Rjk(1, 1);
        Rjk(0, 0) = R(j, k);
        Rjk.gemm(TrQj, Ak, 1, 1); //Rjk = Q*j^T x A*k
        R(j, k) = Rjk(0, 0);
        Ak.gemm(Qj, Rjk, -1, 1); //A*k = A*k - Q*j x Rjk
        for(int i=0; i<dim[0]; i++) {
          (*this)(i, k) = Ak(i, 0);
        }
      }
    }
  }
}
