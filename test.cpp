#include <boost/any.hpp>
#include <cassert>
#include "dense.h"
#include "grid.h"
#include <iostream>
#include "node.h"
#include <vector>

using namespace hicma;

int main(int argc, char** argv) {
  int N = 64;
  int Nb = 4;
  int Nc = N / Nb;
  Grid x(Nc);
  Dense D(Nb);
  D[0] = 3;
  x[0] = D;
  std::cout << boost::any_cast<Dense>(x[0])[0] << '\n';
}
