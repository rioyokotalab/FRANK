#include "hicma/util/l2_error.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/node.h"
#include "hicma/operations/misc/norm.h"
#include "hicma/util/print.h"

#include "yorel/yomm2/cute.hpp"

#include <cmath>


namespace hicma
{

// TODO Inefficient! Avoid copies!
double l2_error(const Node& A, const Node& B) { return l2_error_omm(A, B); }

define_method(double, l2_error_omm, (const Dense& A, const Dense& B)) {
  double diff = norm(A - B);
  double l2 = norm(A);
  return std::sqrt(diff/l2);
}

define_method(double, l2_error_omm, (const Dense& A, const Node& B)) {
  return l2_error(A, Dense(B));
}

define_method(double, l2_error_omm, (const Node& A, const Dense& B)) {
  return l2_error(Dense(A), B);
}

define_method(double, l2_error_omm, (const Node& A, const Node& B)) {
  return l2_error(Dense(A), Dense(B));
}

} // namespace hicma
