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
  int N = 8;
  int Nb = 4;
  int Nc = N / Nb;
  std::vector<double> randx(N);
  Hierarchical A(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  Hierarchical R(Nc, Nc);
  for(int i = 0; i < N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  for(int ic = 0; ic < Nc; ic++) {
    for(int jc = 0; jc < Nc; jc++) {
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      A(ic, jc) = Aij;
      //Fill R with zeros
      Dense Rij(Nb, Nb);
      R(ic, jc) = Rij;
    }
  }
  A.print();
  Hierarchical _A(A);
  //Perform MGS on Block Matrix
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
    //Copy Dense Qsj to Q
    Hierarchical HQsj(DQsj, Nc, 1);
    for(int i = 0; i < Nc; i++) {
      Q(i, j) = HQsj(i, 0);
    }
    //Process next columns
    for(int k = j + 1; k < Nc; k++) {
      Hierarchical HAsk(Nc, 1);
      for(int i = 0; i < Nc; i++) {
        HAsk(i, 0) = A(i, k);
      }
      Dense DAsk(HAsk);
      Dense DRjk(Nb, Nb);
      DRjk.gemm(DQsj, DAsk, CblasTrans, CblasNoTrans, 1, 1); //Rjk = Qsj^T x Ask
      DAsk.gemm(DQsj, DRjk, -1, 1); //A*k = A*k - Q*j x Rjk
      //Copy
      R(j, k) = DRjk;
      Hierarchical _HAsk(DAsk, Nc, 1);
      for(int i = 0; i < Nc; i++) {
        A(i, k) = _HAsk(i, 0);
      }
    }
  }
  print("A");
  _A.print();
  print("Q");
  Q.print();
  print("R");
  R.print();
  Dense DQ(Q);
  Dense DR(R);
  Dense DQR(N, N);
  DQR.gemm(DQ, DR, 1, 1);
  Hierarchical QR(DQR, Nc, Nc);
  print("QR");
  QR.print();
  Dense DA(_A);
  double diff = (DA - DQR).norm();
  double norm = DQR.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}


