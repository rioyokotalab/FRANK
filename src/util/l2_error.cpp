#include "hicma/util/l2_error.h"
#include "hicma/extension_headers/util.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/operations/arithmetic.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/print.h"

#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>


namespace hicma
{

std::tuple<double, double> collect_diff_norm(const Matrix& A, const Matrix& B) {
  return collect_diff_norm_omm(A, B);
}

double l2_error(const Matrix& A, const Matrix& B) {
  assert(get_n_rows(A) == get_n_rows(B));
  assert(get_n_cols(A) == get_n_cols(B));
  double diff, mat_norm;
  std::tie(diff, mat_norm) = collect_diff_norm(A, B);
  //return std::sqrt(diff);
  return std::sqrt(diff/mat_norm);
}

template<typename T>
DoublePair collect_diff_norm_dense_dense(const Dense<T>& A, const Dense<T>& B) {
  double diff = norm(A - B);
  double mat_norm = norm(A);
  return {diff, mat_norm};
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Dense<double>& A, const Dense<double>& B)
) {
  return collect_diff_norm_dense_dense(A, B);
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Dense<float>& A, const Dense<float>& B)
) {
  return collect_diff_norm_dense_dense(A, B);
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Dense<double>& A, const LowRank<double>& B)
) {
  return collect_diff_norm_omm(A, Dense<double>(B));
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Dense<float>& A, const LowRank<float>& B)
) {
  return collect_diff_norm_omm(A, Dense<float>(B));
}

define_method(
  DoublePair, collect_diff_norm_omm, (const LowRank<double>& A, const Dense<double>& B)
) {
  return collect_diff_norm_omm(Dense<double>(A), B);
}

define_method(
  DoublePair, collect_diff_norm_omm, (const LowRank<float>& A, const Dense<float>& B)
) {
  return collect_diff_norm_omm(Dense<float>(A), B);
}

define_method(
  DoublePair, collect_diff_norm_omm, (const LowRank<double>& A, const LowRank<double>& B)
) {
  return collect_diff_norm_omm(Dense<double>(A), Dense<double>(B));
}

define_method(
  DoublePair, collect_diff_norm_omm, (const LowRank<float>& A, const LowRank<float>& B)
) {
  return collect_diff_norm_omm(Dense<float>(A), Dense<float>(B));
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Hierarchical<double>& A, const Matrix& B)
) {
  Hierarchical<double> BH = split<double>(B, A.dim[0], A.dim[1]);
  return collect_diff_norm_omm(A, BH);
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Hierarchical<float>& A, const Matrix& B)
) {
  Hierarchical<float> BH = split<float>(B, A.dim[0], A.dim[1]);
  return collect_diff_norm_omm(A, BH);
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Matrix& A, const Hierarchical<double>& B)
) {
  Hierarchical<double> AH = split<double>(A, B.dim[0], B.dim[1]);
  return collect_diff_norm_omm(AH, B);
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Matrix& A, const Hierarchical<float>& B)
) {
  Hierarchical<float> AH = split<float>(A, B.dim[0], B.dim[1]);
  return collect_diff_norm_omm(AH, B);
}

define_method(
  DoublePair, collect_diff_norm_omm,
  (const Hierarchical<double>& A, const Hierarchical<double>& B)
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
    return collect_diff_norm_omm(Dense<double>(A), Dense<double>(B));
  }
}

define_method(
  DoublePair, collect_diff_norm_omm,
  (const Hierarchical<float>& A, const Hierarchical<float>& B)
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
    return collect_diff_norm_omm(Dense<float>(A), Dense<float>(B));
  }
}

define_method(
  DoublePair, collect_diff_norm_omm, (const Matrix& A, const Matrix& B)
) {
  omm_error_handler("collect_diff_norm", {A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
