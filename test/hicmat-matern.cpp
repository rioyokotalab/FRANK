#include "hicma/hicma.h"

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>
#include <iostream>
#include <fstream>

#include <starsh.h>

using namespace hicma;
using namespace std;

int main(int argc, char** argv) {
  timing::start("Overall");
  hicma::initialize();
  
  int64_t nleaf = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t N = atoi(argv[3]);
  int64_t admis = atoi(argv[4]);
  int64_t basis = 0;
  int64_t nblocks = N / nleaf;
  assert(basis == NORMAL_BASIS || basis == SHARED_BASIS);

  /* Default parameters for statistics */
  double beta = 0.1;
  double nu = 0.5;//in matern, nu=0.5 exp (half smooth), nu=inf sqexp (inifinetly smooth)
  //nu is only used in matern kernel
  //double noise = 1.e-2; // did not work for 10M in Lorapo
  double noise = 1.e-1;
  double sigma = 1.0;
  double wave_k = 1.0;
  int add_diag = 27000;
  starsh::matern_kernel_prepare(N, beta, nu, noise, sigma, add_diag);  

  std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};
  
  Dense x(random_uniform, std::vector<std::vector<double>>(), N);
  Dense b(N);
  Dense D(starsh::matern_kernel_fill, randx, N, N);


  print("Being compression");
  timing::start("Hierarchical compression");
  Hierarchical A(starsh::matern_kernel_fill, randx, N, N, rank, nleaf, admis,
               nblocks, nblocks, basis);
  timing::stop("Hierarchical compression");
  printXML(A, std::to_string(N) + std::string("-") + std::to_string(nleaf) + std::string("-") +
           std::to_string(rank) + std::string("-") + std::to_string(admis) + std::string(".xml"));

  print("Compression Accuracy");
  timing::start("Compression accuracy check");
  double comp_error = l2_error(A, D);
  double comp_rate = double(get_memory_usage(D)) / double(get_memory_usage(A));
  timing::stop("Compression accuracy check");
  print("Rel. L2 Error", comp_error, false);

  gemm(A, x, b, 1, 1);

  timing::start("LU decomposition");
  Hierarchical L, U;
  start_schedule();
  std::tie(L, U) = getrf(A);
  execute_schedule();
  timing::stopAndPrint("LU decomposition", 2);

  timing::start("Solution");
  trsm(L, b, TRSM_LOWER);
  trsm(U, b, TRSM_UPPER);
  timing::stopAndPrint("Solution");

  double solve_acc = l2_error(x, b);
  print("LU Accuracy");
  print("Rel. L2 Error", solve_acc, false);

  std::ofstream file;
  file.open("acc-result.csv", std::ios::app | std::ios::out);
  file << N << "," << nleaf << "," << rank << "," << admis << "," << comp_error << "," << comp_rate << ","  << solve_acc <<  std::endl;
  file.close();
  
  return 0;
}
