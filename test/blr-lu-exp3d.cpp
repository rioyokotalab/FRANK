#include "hicma/hicma.h"

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>
#include <iostream>
#include <fstream>

using namespace hicma;
using namespace std;

int main(int, char** argv) {
  timing::start("Overall");
  hicma::initialize();

  int64_t nleaf = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t N = atoi(argv[3]);
  int64_t admis = atoi(argv[4]);
  int64_t nblocks = N / nleaf;

  /* Default parameters for statistics */
  double beta = 0.1;
  double nu = 0.5;//in matern, nu=0.5 exp (half smooth), nu=inf sqexp (inifinetly smooth)
  double noise = 1.e-1;
  double sigma = 1.0;

  starsh::exp_kernel_prepare(N, beta, nu, noise, sigma, 3);

  std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};

  Dense x(random_uniform, std::vector<std::vector<double>>(), N);
  Dense b(N);

  print("Being compression");
  timing::start("Hierarchical compression");
  start_schedule();
  Hierarchical A(starsh::exp_kernel_fill, randx, N, N, rank, nleaf, admis,
               nblocks, nblocks);
  execute_schedule();
  double comp_time = timing::stop("Hierarchical compression");

  gemm(A, x, b, 1, 1);

  timing::start("LU decomposition");
  Hierarchical L, U;
  start_schedule();
  std::tie(L, U) = getrf(A);
  execute_schedule();
  double fact_time = timing::stop("LU decomposition");

  timing::start("Solution");
  trsm(L, b, TRSM_LOWER);
  trsm(U, b, TRSM_UPPER);
  timing::stopAndPrint("Solution");

  double solve_acc = l2_error(x, b);
  print("LU Accuracy");
  print("Rel. L2 Error", solve_acc, false);

  std::ofstream file;
  file.open("blr-lu-exp3d.csv", std::ios::app | std::ios::out);
  file << N << "," << nleaf << "," << rank << "," << admis << ","  << solve_acc
       << "," << fact_time << "," << comp_time << ","  << std::getenv("STARPU_NCPU") <<  std::endl;
  file.close();

  return 0;
}
