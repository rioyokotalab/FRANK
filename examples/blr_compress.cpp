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
  if(inputName.length() == 0) { //Default to Laplace1D kernel
    nodes.push_back(equallySpacedVector(N, 0.0, 1.0));
    D = Hierarchical<double>(LaplacendKernel<double>(nodes), N, N, Nb, Nb, Nc, Nc, Nc);
    A = Hierarchical<double>(LaplacendKernel<double>(nodes), N, N, rank, Nb, admis, Nc, Nc, POSITION_BASED_ADMIS);
    outName <<"Laplace1D_"<<N;
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
  // outName <<"_"<<Nb;
  // write_JSON(D, outName.str() + "_dense.json");
  // outName <<"_" <<admis;
  // write_JSON(A, outName.str() + ".json");
  
  return 0;
}

