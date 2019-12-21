#include "hicma/operations/misc/norm.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"

#include "yorel/multi_methods.hpp"

namespace hicma
{

double norm(const Node& A) {
  return norm_omm(A);
}

BEGIN_SPECIALIZATION(norm_omm, double, const Dense& A) {
  double l2 = 0;
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<A.dim[1]; j++) {
      l2 += A(i, j) * A(i, j);
    }
  }
  return l2;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(norm_omm, double, const LowRank& A) {
  return norm(Dense(A));
} END_SPECIALIZATION;


BEGIN_SPECIALIZATION(norm_omm, double, const Hierarchical& A) {
  double l2 = 0;
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<A.dim[1]; j++) {
      l2 += norm(A(i, j));
    }
  }
  return l2;
} END_SPECIALIZATION;


BEGIN_SPECIALIZATION(norm_omm, double, const Node& A) {
  std::cerr << "norm(";
  std::cerr << A.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
