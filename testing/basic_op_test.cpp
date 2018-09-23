#include <algorithm>
#include "mpi_utils.h"
#include "functions.h"
#include "print.h"
#include "timer.h"
#include "hierarchical.h"
#include "dense.h"
#include "node.h"

using namespace hicma;

int main(int argc, char** argv) {
  Hierarchical A(2, 2);
  Dense B(2, 2);
  Dense C = B;
  B.reset();
  std::cout << C.use_count() << std::endl;
  std::cout << "From main: " << C.is_string() << std::endl;

  A[0] = C;
  C.reset();
  std::cout << "From main: " << A[0].is_string() << std::endl;
}
