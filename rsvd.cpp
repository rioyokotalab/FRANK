#include "node.h"
#include "dense.h"
#include <fstream>
#include <iomanip>

using namespace hicma;

int main(int argc, char** argv) {
  const int N = 8;
  const int k = 4;
  Dense A(N,N);
  Dense G(N,k+5);
  double sum = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A(i,j) = 1. / std::abs(i-j-3*N);
      sum += A(i,j);
    }
  }
  std::fstream file("rand.dat");
  for (int i=0; i<k; i++) {
    for (int j=0; j<N; j++) {
      file >> G(j,i);
    }
  }
  file.close();
  Dense Y = A * G;
  std::cout << std::setprecision(16) << Y(0,1) << std::endl;
  return 0;
}
