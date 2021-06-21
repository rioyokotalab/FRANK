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
  std::stringstream outName;

  Hierarchical A,D;
  std::vector<std::vector<double>> nodes;
  if(inputName.length() == 0) { //Default to Laplace2D kernel
    nodes.push_back(equallySpacedVector(N, 0.0, 1.0));
    nodes.push_back(equallySpacedVector(N, 0.0, 1.0));
    D = Hierarchical(laplacend, nodes, N, N, Nb, Nb, Nc, Nc, Nc);
    A = Hierarchical(laplacend, nodes, N, N, rank, Nb, admis, Nc, Nc, POSITION_BASED_ADMIS);
    outName <<"Laplace2D_"<<N;
  }
  else if(inputName == "starsh") { // 3D Exponential kernel from Stars-H
    /* Temporarily disabled
    //Use starsh 3D exponential kernel
    //Default parameters for statistics
    double beta = 0.1;
    double nu = 0.5;//in matern, nu=0.5 exp (half smooth), nu=inf sqexp (inifinetly smooth)
    double noise = 1.e-1;
    double sigma = 1.0;
    int ndim = 3;
    D = Hierarchical(N, Nb, Nc, beta, nu, noise, sigma, ndim, (double)Nc, Nb, NORMAL_BASIS, POSITION_BASED_ADMIS);
    A = Hierarchical(N, Nb, Nc, beta, nu, noise, sigma, ndim, admis, rank, NORMAL_BASIS, GEOMETRY_BASED_ADMIS);

    std::vector<std::vector<double>> randx{get_sorted_random_vector(N)};
    starsh::exp_kernel_prepare(N, beta, nu, noise, sigma, ndim);
    D = Hierarchical(starsh::exp_kernel_fill, randx, N, N, Nb, Nb, (double)Nc);
    A = Hierarchical(starsh::exp_kernel_fill, randx, N, N, rank, Nb, admis);
    outName <<"Stars-H_Exponential_3D_"<<N;
    */
    std::cout <<"Stars-H kernels are temporarily disabled" <<std::endl;
    return 0;
  }
  else { // Read matrix (.csv) and geometry information (.geom)
    nodes = read_geometry_file(inputName+".geom");
    D = Hierarchical(inputName+".csv", HICMA_ROW_MAJOR, nodes, N, N, 0, Nb, Nc, Nc, Nc);
    A = Hierarchical(inputName+".csv", HICMA_ROW_MAJOR, nodes, N, N, rank, Nb, admis, Nc, Nc, GEOMETRY_BASED_ADMIS);
    outName <<inputName<<"_"<<N;
  }
  print("BLR Compression Accuracy");
  print("Rel. L2 Error", l2_error(D, A), false);

  //Write matrices to JSON file
  outName <<"_"<<Nb;
  write_JSON(D, outName.str() + "_dense.json");
  outName <<"_" <<admis;
  write_JSON(A, outName.str() + ".json");
  
  return 0;
}

