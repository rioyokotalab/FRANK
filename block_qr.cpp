#include "any.h"
#include "low_rank.h"
#include "hierarchical.h"
#include "functions.h"
#include "batch.h"
#include "print.h"
#include "timer.h"

#include <algorithm>
#include <cmath>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 64;
  int Nb = 16;
  int Nc = N / Nb;
  std::vector<double> randx(N);
  Hierarchical A(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  Hierarchical R(Nc, Nc);
  for(int i = 0; i < N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  print("Time");
  start("Init matrix");
  for(int ic = 0; ic < Nc; ic++) {
    for(int jc = 0; jc < Nc; jc++) {
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      A(ic, jc) = Aij;
      //Fill R with zeros
      Dense Rij(Nb, Nb);
      R(ic, jc) = Rij;
    }
  }
  stop("Init matrix");
  Hierarchical _A(A); //Copy of A
  start("QR decomposition");
  for(int j = 0; j < Nc; j++) {
    Hierarchical HAsj(Nc, 1);
    for(int i = 0; i < Nc; i++) {
      HAsj(i, 0) = A(i, j);
    }
    Dense DAsj(HAsj);
    Dense DQsj(DAsj.dim[0], DAsj.dim[1]);
    Dense Rjj(Nb, Nb);
    DAsj.qr(DQsj, Rjj); //[Q*j, Rjj] = QR(A*j)
    R(j, j) = Rjj;
    //Copy Dense Qsj to Hierarchical Q
    Hierarchical HQsj(DQsj, Nc, 1);
    for(int i = 0; i < Nc; i++) {
      Q(i, j) = HQsj(i, 0);
    }
    //Process next columns
    for(int k = j + 1; k < Nc; k++) {
      //Take k-th column
      Hierarchical HAsk(Nc, 1);
      for(int i = 0; i < Nc; i++) {
        HAsk(i, 0) = A(i, k);
      }
      Dense DAsk(HAsk);
      Dense DRjk(Nb, Nb);
      DRjk.gemm(DQsj, DAsk, CblasTrans, CblasNoTrans, 1, 1); //Rjk = Qsj^T x Ask
      R(j, k) = DRjk;
      DAsk.gemm(DQsj, DRjk, -1, 1); //A*k = A*k - Q*j x Rjk
      Hierarchical _HAsk(DAsk, Nc, 1);
      for(int i = 0; i < Nc; i++) {
        A(i, k) = _HAsk(i, 0);
      }
    }
  }
  stop("QR decomposition");
  printTime("-DGEQRF");
  printTime("-DGEMM");
  Dense QR(N, N);
  QR.gemm(Dense(Q), Dense(R), 1, 1);
  double diff = (Dense(_A) - QR).norm();
  double norm = _A.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}

