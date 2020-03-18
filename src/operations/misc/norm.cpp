#include "hicma/operations/misc/norm.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"

namespace hicma
{

double norm(const Node& A) {
  return norm_omm(A);
}

define_method(double, norm_omm, (const Dense& A)) {
  double l2 = 0;
  timing::start("Norm(Dense)");
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<A.dim[1]; j++) {
      l2 += A(i, j) * A(i, j);
    }
  }
  timing::stop("Norm(Dense)");
  return l2;
}

define_method(double, norm_omm, (const LowRank& A)) {
  return norm(Dense(A));
}

define_method(double, norm_omm, (const Hierarchical& A)) {
  double l2 = 0;
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<A.dim[1]; j++) {
      l2 += norm(A(i, j));
    }
  }
  return l2;
}

define_method(double, norm_omm, (const Node& A)) {
  std::cerr << "norm(";
  std::cerr << A.type();
  std::cerr << ") undefined." << std::endl;
  abort();
}

} // namespace hicma
