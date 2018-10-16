#include "functions.h"
#include "low_rank.h"
#include "timer.h"

using namespace hicma;

int main(int argc, char** argv) {
  int N = 32;
  int rank = 16;
  std::vector<double> randx(2*N);
  for (int i=0; i<2*N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());
  print("Time");
  start("Init matrix");
  Dense D(laplace1d, randx, N, N-2, 0, N);
  stop("Init matrix");
  start("Randomized SVD");
  LowRank LR(D, rank);
  stop("Randomized SVD");
  double diff = (D - Dense(LR)).norm();
  double norm = D.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
