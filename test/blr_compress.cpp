#include "hicma/hicma.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <string>

using namespace hicma;

int main(int argc, char** argv) {
  hicma::initialize();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t Nb = argc > 2 ? atoi(argv[2]) : 32;
  int64_t Nc = N / Nb;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 16;
  double admis = argc > 4 ? atof(argv[4]) : 0;
  std::string inputName = argc > 5 ? std::string(argv[5]) : "";

  Hierarchical A,D;
  std::vector<std::vector<double>> nodes;
  if(inputName.length() > 0) { //Supply inputName.csv and inputName.geom files
    nodes = read_geometry_file(inputName+".geom");
    D = Hierarchical(inputName+".csv", HICMA_ROW_MAJOR, nodes, N, N, 0, Nb, Nc, Nc, Nc);
    A = Hierarchical(inputName+".csv", HICMA_ROW_MAJOR, nodes, N, N, rank, Nb, admis, Nc, Nc, NORMAL_BASIS, GEOMETRY_BASED_ADMIS);
  }
  else { //Use starsh 3D exponential kernel
    /* Default parameters for statistics */
    double beta = 0.1;
    double nu = 0.5;//in matern, nu=0.5 exp (half smooth), nu=inf sqexp (inifinetly smooth)
    double noise = 1.e-1;
    double sigma = 1.0;
    int ndim = 3;
    D = Hierarchical(N, Nb, Nc, beta, nu, noise, sigma, ndim, (double)Nc, Nb, NORMAL_BASIS, POSITION_BASED_ADMIS);
    A = Hierarchical(N, Nb, Nc, beta, nu, noise, sigma, ndim, admis, rank, NORMAL_BASIS, GEOMETRY_BASED_ADMIS);

    // std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};
    // starsh::exp_kernel_prepare(N, beta, nu, noise, sigma, ndim);
    // D = Hierarchical(starsh::exp_kernel_fill, randx, N, N, Nb, Nb, (double)Nc);
    // A = Hierarchical(starsh::exp_kernel_fill, randx, N, N, rank, Nb, admis);
  }
  print("Compression with BLR-matrix");
  std::cout <<inputName <<", N=" <<N <<",b=" <<Nb <<",rank=" <<rank <<",admis=" <<admis <<std::endl;
  print("Rel. L2 Error", l2_error(A, D), false);

  //Output matrix as XML
  // std::stringstream outName;
  // if(inputName.length () > 0)
  //   outName <<inputName;
  // else
  //   outName <<"starsh_exp3d_" <<N;
  // outName <<"_" <<Nb;
  // printXML(D, outName.str() + ".xml");
  // outName <<"_" <<admis;
  // printXML(A, outName.str() + ".xml");

  return 0;
}

