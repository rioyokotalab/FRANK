#include "FRANK/util/l2_error.h"

#include "FRANK/definitions.h"
#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/operations/arithmetic.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"
#include "FRANK/util/print.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>


namespace FRANK
{

declare_method(
  DoublePair, collect_diff_norm_omm,
  (virtual_<const Matrix&>, virtual_<const Matrix&>)
)

std::tuple<double, double> collect_diff_norm(const Matrix& A, const Matrix& B) {
  return collect_diff_norm_omm(A, B);
}

double l2_error(const Matrix& A, const Matrix& B) {
  assert(get_n_rows(A) == get_n_rows(B));
  assert(get_n_cols(A) == get_n_cols(B));
  double diff, mat_norm;
  std::tie(diff, mat_norm) = collect_diff_norm(A, B);
  return std::sqrt(diff/(mat_norm + std::numeric_limits<double>::epsilon()));
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Dense& A, const Dense& B)
) {
  const double diff = norm(A - B);
  const double mat_norm = norm(A);
  return {diff, mat_norm};
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Dense& A, const LowRank& B)
) {
  return collect_diff_norm_omm(A, Dense(B));
}

define_method(
  DoublePair, collect_diff_norm_omm, (const LowRank& A, const Dense& B)
) {
  return collect_diff_norm_omm(Dense(A), B);
}

define_method(
  DoublePair, collect_diff_norm_omm, (const LowRank& A, const LowRank& B)
) {
  return collect_diff_norm_omm(Dense(A), Dense(B));
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Hierarchical& A, const Matrix& B)
) {
  const Hierarchical BH = split(B, A.dim[0], A.dim[1]);
  return collect_diff_norm_omm(A, BH);
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Matrix& A, const Hierarchical& B)
) {
  const Hierarchical AH = split(A, B.dim[0], B.dim[1]);
  return collect_diff_norm_omm(AH, B);
}

define_method(
  DoublePair, collect_diff_norm_omm,
  (const Hierarchical& A, const Hierarchical& B)
) {
  if (A.dim[0] == B.dim[0] && A.dim[1] == B.dim[1]) {
    double total_diff = 0, total_norm = 0;
    for (int64_t i=0; i<A.dim[0]; ++i) {
      for (int64_t j=0; j<A.dim[1]; ++j) {
        double diff, mat_norm;
        std::tie(diff, mat_norm) = collect_diff_norm_omm(A(i, j), B(i, j));
        total_diff += diff;
        total_norm += mat_norm;
      }
    }
    return {total_diff, total_norm};
  } else {
    return collect_diff_norm_omm(Dense(A), Dense(B));
  }
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Matrix& A, const Matrix& B)
) {
  omm_error_handler("collect_diff_norm", {A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
