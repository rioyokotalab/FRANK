#include "hicma/hicma.h"

#include <algorithm>
#include <cmath>

#include "yorel/multi_methods.hpp"

using namespace hicma;

int main() {
  yorel::multi_methods::initialize();
  int N = 64;
  int Nb = 16;
  int Nc = N / Nb;
  std::vector<double> randx = get_sorted_random_vector(N);
  Hierarchical A(Nc, Nc);
  Hierarchical Q(Nc, Nc);
  Hierarchical R(Nc, Nc);
  print("Time");
  timing::start("Init matrix");
  for(int ic = 0; ic < Nc; ic++) {
    for(int jc = 0; jc < Nc; jc++) {
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      A(ic, jc) = Aij;
      //Fill R with zeros
      Dense Rij(Nb, Nb);
      R(ic, jc) = Rij;
    }
  }
  timing::stopAndPrint("Init matrix");
  Hierarchical _A(A); //Copy of A
  timing::start("QR decomposition");
  for(int j = 0; j < Nc; j++) {
    Hierarchical HAsj(Nc, 1);
    for(int i = 0; i < Nc; i++) {
      HAsj(i, 0) = A(i, j);
    }
    Dense DAsj(HAsj);
    Dense DQsj(DAsj.dim[0], DAsj.dim[1]);
    Dense Rjj(Nb, Nb);
    qr(DAsj, DQsj, Rjj); //[Q*j, Rjj] = QR(A*j)
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
      gemm(DQsj, DAsk, DRjk, true, false, 1, 1); //Rjk = Qsj^T x Ask
      R(j, k) = DRjk;
      gemm(DQsj, DRjk, DAsk, -1, 1); //A*k = A*k - Q*j x Rjk
      Hierarchical _HAsk(DAsk, Nc, 1);
      for(int i = 0; i < Nc; i++) {
        A(i, k) = _HAsk(i, 0);
      }
    }
  }
  timing::stopAndPrint("QR decomposition");
  Dense QR(N, N);
  gemm(Dense(Q), Dense(R), QR, 1, 1);
  print("Accuracy");
  print("Rel. L2 Error", l2_error(_A, QR), false);
  return 0;
}


