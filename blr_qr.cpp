#include "any.h"
#include "low_rank.h"
#include "hierarchical.h"
#include "functions.h"
#include "batch.h"
#include "print.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 64;
  int Nb = 16;
  int Nc = N / Nb;
  int rank = 8;
  std::vector<double> randx(N);
  Hierarchical A(Nc, Nc);
  Hierarchical D(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  Hierarchical R(Nc, Nc);
  for(int i = 0; i < N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      D(ic,jc) = Aij;
      if (std::abs(ic - jc) <= 1) {
        A(ic,jc) = Aij;
      }
      else {
        rsvd_push(A(ic,jc), Aij, rank);
      }
      //Fill R with zeros
      Dense Rij(Nb, Nb);
      R(ic, jc) = Rij;
    }
  }
  rsvd_batch();
  double diff = 0, norm = 0;
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      if(A(ic,jc).is(HICMA_LOWRANK)) {
        diff += (Dense(A(ic,jc)) - D(ic,jc)).norm();
        norm += D(ic,jc).norm();
      }
    }
  }
  print("Compression Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  print("Time");
  Hierarchical _A(A); //Copy of A
  start("QR decomposition");
  for(int j = 0; j < Nc; j++) {
    Hierarchical Qu(Nc, 1);
    Hierarchical Ru(Nc, 1);
    Hierarchical Bsj(Nc, 1);
    for(int i = 0; i < Nc; i++) {
      if(A(i, j).is(HICMA_DENSE)) {
        // [Qui, Rui] = qr(A(i, j).U)
        Dense Qui(Nb, Nb);
        Dense Rui(Nb, Nb);
        for(int k = 0; k < Nb; k++) {
          Qui(k, k) = 1.0;
          Rui(k, k) = 1.0;
        }
        // If A(i, j) is dense, assume A(i, j).U = I
        Qu(i, 0) = Qui;
        Ru(i, 0) = Rui;

        // Bsj(i, 0) = Rui * A(i, j).S * A(i, j).V
        // If A(i, j) is dense, Rui = A(i, j).S = I, and A(i, j).V = A(i, j).
        Bsj(i, 0) = A(i, j);
      }
      else if(A(i, j).is(HICMA_LOWRANK)) {
        LowRank Aij(A(i, j));
        // [Qui, Rui] = qr(A(i, j).U)
        int mu = Aij.U.dim[0];
        int nu = Aij.U.dim[1];
        Dense Qui(mu, nu);
        Dense Rui(nu, nu);
        Dense AUij(Aij.U);
        AUij.qr(Qui, Rui);
        Qu(i, 0) = Qui;
        Ru(i, 0) = Rui;

        // Bsj(i, 0) = Rui * A(i, j).S * A(i, j).V
        Dense RS(Rui.dim[0], Aij.S.dim[1]);
        RS.gemm(Rui, Aij.S, 1, 1);
        Dense RSV(RS.dim[0], Aij.V.dim[1]);
        RSV.gemm(RS, Aij.V, 1, 1);
        Bsj(i, 0) = RSV;
      }
    }
    Dense DBsj(Bsj);
    Dense Qb(DBsj.dim[0], DBsj.dim[1]);
    Dense Rb(DBsj.dim[1], DBsj.dim[1]);
    DBsj.qr(Qb, Rb); //[Qb, Rb] = qr(Bsj);
    R(j, j) = Rb; //Rjj = Rb
    //Slice Qb by row based on Bsj
    Hierarchical HQb(Nc, 1);
    int rowOffset = 0;
    for(int i = 0; i < Nc; i++) {
      Dense Bij(Bsj(i, 0));
      Dense HQBij(Bij.dim[0], Bij.dim[1]);
      for(int row = 0; row < Bij.dim[0]; row++) {
        for(int col = 0; col < Bij.dim[1]; col++) {
          HQBij(row, col) = Qb(rowOffset + row, col);
        }
      }
      HQb(i, 0) = HQBij;
      rowOffset += Bij.dim[0];
    }
    //Q*j = Qu * Qb
    Hierarchical Qsj(Nc, 1);
    for(int i = 0; i < Nc; i++) {
      Dense DQuij(Qu(i, 0));
      Dense DQbij(HQb(i, 0));
      Dense res(DQuij.dim[0], DQbij.dim[1]);
      res.gemm(DQuij, DQbij, 1, 1);
      Qsj(i, 0) = res;
    }
    //Copy column of Qsj to Q
    for(int i = 0; i < Nc; i++) {
      Q(i, j) = Qsj(i, 0);
    }
    //Form transpose of Qsj to be used in processing next columns
    Hierarchical TrQsj(1, Nc);
    for(int i = 0; i < Nc; i++) {
      if(Qsj(i, 0).is(HICMA_DENSE)) {
        Dense Qsij(Qsj(i, 0));
        Qsij.transpose();
        TrQsj(0, i) = Qsij;
      }
      else if(Qsj(i, 0).is(HICMA_LOWRANK)) {
        LowRank Qsij(Qsj(i, 0));
        Qsij.transpose();
        TrQsj(0, i) = Qsij;
      }
    }
    //Process next columns
    for(int k = j + 1; k < Nc; k++) {
      //Take k-th column
      Hierarchical HAsk(Nc, 1);
      for(int i = 0; i < Nc; i++) {
        HAsk(i, 0) = A(i, k);
      }
      Dense DRjk(Nb, Nb);
      DRjk.gemm(TrQsj, HAsk, 1, 1); //Rjk = Q*j^T x A*k
      LowRank Rjk(DRjk, rank); 
      R(j, k) = Rjk;

      HAsk.gemm(Qsj, Rjk, -1, 1); //A*k = A*k - Q*j x Rjk
      for(int i = 0; i < Nc; i++) {
        A(i, k) = HAsk(i, 0);
      }
    }
  }
  stop("QR decomposition");
  printTime("-DGEQRF");
  printTime("-DGEMM");
  Dense QR(N, N);
  QR.gemm(Dense(Q), Dense(R), 1, 1);
  diff = (Dense(_A) - QR).norm();
  norm = QR.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}


