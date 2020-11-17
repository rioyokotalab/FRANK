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
    D = Hierarchical(inputName+".csv", MATRIX_ROW_MAJOR, nodes, N, N, 0, Nb, Nc, Nc, Nc);
    A = Hierarchical(inputName+".csv", MATRIX_ROW_MAJOR, nodes, N, N, rank, Nb, admis, Nc, Nc, NORMAL_BASIS, GEOMETRY_BASED_ADMIS);
  }
  else { //Use kernel and generated points
    nodes.push_back(equallySpacedVector(N, 0.0, 1.0)); //x1,x2,...,xn
    nodes.push_back(equallySpacedVector(N, 0.0, 1.0)); //y1,y2,...,yn
    nodes.push_back(equallySpacedVector(N, 0.0, 1.0)); //z1,z2,...,zn
    D = Hierarchical(laplacend, nodes, N, N, 0, Nb, Nc, Nc, Nc);
    A = Hierarchical(laplacend, nodes, N, N, rank, Nb, admis, Nc, Nc, NORMAL_BASIS, GEOMETRY_BASED_ADMIS);
  }
  print("Compression with BLR-matrix");
  std::cout <<inputName <<", N=" <<N <<",b=" <<Nb <<",rank=" <<rank <<",admis=" <<admis <<std::endl;
  print("Rel. L2 Error", l2_error(D, A), false);

  std::stringstream outName;
  if(inputName.length () > 0)
    outName <<inputName;
  else
    outName <<"Laplace3D_" <<N;
  outName <<"_" <<Nb <<"_" <<admis <<".xml";
  printXML(A, outName.str());
  return 0;
}

