#include "functions.h"
#include "low_rank.h"
#include "timer.h"

using namespace hicma;

int main(int argc, char** argv) {
  int N = 32;
  int k = 16;
  std::vector<double> randx(2*N);
  for (int i=0; i<2*N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  print("Time");
  start("Init matrix");
  Dense A(laplace1d, randx, N, N-2, 0, N);
  stop("Init matrix");
  start("Randomized SVD");
  LowRank LR(A.dim[0],A.dim[1],k);
  int rank = k + 5;
  LR.S = Dense(rank, rank);
  Dense RN(A.dim[1],rank);
  std::mt19937 generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  for (int i=0; i<A.dim[1]*rank; i++) {
    RN[i] = distribution(generator); // RN = randn(n,k+p)
  }
  Dense Y(A.dim[0],rank);
  Y.gemm(A, RN, CblasNoTrans, CblasNoTrans, 1, 0); // Y = A * RN
  Dense Q(A.dim[0],rank);
  Dense R(rank,rank);
  Y.qr(Q, R); // [Q, R] = qr(Y)
  Dense Bt(A.dim[1],rank);
  Bt.gemm(A, Q, CblasTrans, CblasNoTrans, 1, 0); // B' = A' * Q
  Dense Qb(A.dim[1],rank);
  Dense Rb(rank,rank);
  Bt.qr(Qb,Rb); // [Qb, Rb] = qr(B')
  Dense Ur(rank,rank);
  Dense Vr(rank,rank);
  Rb.svd(Vr,LR.S,Ur); // [Vr, S, Ur] = svd(Rb);
  Ur.resize(k,rank);
  LR.U.gemm(Q, Ur, CblasNoTrans, CblasTrans, 1, 0); // U = Q * Ur'
  Vr.resize(rank,k);
  LR.V.gemm(Vr, Qb, CblasTrans, CblasTrans, 1, 0); // V = Vr' * Qb'
  LR.S.resize(k,k);
  stop("Randomized SVD");
  double diff = (A - Dense(LR)).norm();
  double norm = A.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
