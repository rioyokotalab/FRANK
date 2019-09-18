#include "any.h"
#include "low_rank.h"
#include "functions.h"
#include "print.h"
#include "timer.h"

#include <algorithm>
#include <cmath>

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
  start("LR Add");
  LowRank A(D, rank);
  LowRank B(D, rank);
  A += B;
  stop("LR Add");
  double diff = (D + D - Dense(A)).norm();
  double norm = D.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
