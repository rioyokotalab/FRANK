#include "hicma/util/l2_error.h"

#include "hicma/node.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/operations/norm.h"
#include "hicma/util/print.h"

#include <cmath>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

double l2_error(const Node& A, const Node& B) {
  return l2_error_omm(A, B);
}

BEGIN_SPECIALIZATION(
  l2_error_omm, double,
  const Dense& A, const Dense& B
) {
  double diff = norm(A - B);
  double l2 = norm(A);
  return std::sqrt(diff/l2);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  l2_error_omm, double,
  const Dense& A, const Node& B
) {
  return l2_error(A, Dense(B));
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  l2_error_omm, double,
  const Node& A, const Dense& B
) {
  return l2_error(Dense(A), B);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  l2_error_omm, double,
  const Node& A, const Node& B
) {
  return l2_error(Dense(A), Dense(B));
} END_SPECIALIZATION;

} // namespace hicma
