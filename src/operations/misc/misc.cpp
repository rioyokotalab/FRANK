#include "hicma/operations/misc.h"
#include "hicma/extension_headers/classes.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/util/omm_error_handler.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>


namespace hicma
{

void zero_all(Matrix& A) {
  zero_all_omm(A);
}

define_method(void, zero_all_omm, (Dense& A)) {
  A = 0.0;
}

define_method(void, zero_all_omm, (LowRank& A)) {
  A.U = 0.0;
  A.S = 0.0;
  A.V = 0.0;
  for(int64_t i = 0; i < std::min(A.U.dim[0], A.U.dim[1]); i++) {
    A.U(i, i) = 1;
  }
  for(int64_t i = 0; i < std::min(A.V.dim[0], A.V.dim[1]); i++) {
    A.V(i, i) = 1;
  }
}

define_method(void, zero_all_omm, (Hierarchical& A)) {
  for(int64_t i = 0; i < A.dim[0]; i++)
    for(int64_t j = 0; j < A.dim[1]; j++) {
      zero_all(A(i, j));
    }
}

define_method(void, zero_all_omm, (Matrix& A)) {
  omm_error_handler("zero_all", {A}, __FILE__, __LINE__);
  std::abort();
}


void zero_lower(Matrix& A) {
  zero_lower_omm(A);
}

define_method(void, zero_lower_omm, (Dense& A)) {
  for(int64_t i=0; i<A.dim[0]; i++)
    for(int64_t j=0; j<std::min(i, A.dim[1]); j++)
      A(i,j) = 0.0;
}

define_method(void, zero_lower_omm, (Hierarchical& A)) {
  for(int64_t i=0; i<A.dim[0]; i++)
    for(int64_t j=0; j<=std::min(i, A.dim[1]-1); j++) {
      if(j == i)
        zero_lower(A(i, j));
      else
        zero_all(A(i, j));
    }
}

define_method(void, zero_lower_omm, (Matrix& A)) {
  omm_error_handler("zero_lower", {A}, __FILE__, __LINE__);
  std::abort();
}

void zero_upper(Matrix& A) {
  zero_upper_omm(A);
}

define_method(void, zero_upper_omm, (Dense& A)) {
  for(int64_t i=0; i<A.dim[0]; i++)
    for(int64_t j=i+1; j<A.dim[1]; j++)
      A(i,j) = 0.0;
}

define_method(void, zero_upper_omm, (Hierarchical& A)) {
  for(int64_t i=0; i<A.dim[0]; i++)
    for(int64_t j=i; j<A.dim[1]; j++) {
      if(j == i)
        zero_upper(A(i, j));
      else
        zero_all(A(i, j));
    }
}

define_method(void, zero_upper_omm, (Matrix& A)) {
  omm_error_handler("zero_upper", {A}, __FILE__, __LINE__);
  std::abort();
}


double cond(Dense A) {
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<double> S = get_singular_values(A);
  return (S[0] / S[k-1]);
}

std::vector<double> equallySpacedVector(int64_t N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int64_t i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

int64_t find_svd_truncation_rank(const Dense& S, double eps) {
  double threshold = eps * std::sqrt(norm(S));
  int64_t min_dim = std::min(S.dim[0], S.dim[1]);
  int64_t rank = 0;
  double err = 0;
  do {
    rank++;
    err = 0.0;
    for(int64_t j=rank; j<min_dim; j++) {
      err += S(rank, rank)*S(rank, rank);
    }
  } while(rank < min_dim && std::sqrt(err) > threshold);
  return rank;
}

Hierarchical split(
  const Matrix& A, int64_t n_row_blocks, int64_t n_col_blocks, bool copy
) {
  return split_omm(
    A,
    IndexRange(0, get_n_rows(A)).split(n_row_blocks),
    IndexRange(0, get_n_cols(A)).split(n_col_blocks),
    copy
  );
}

Hierarchical split(const Matrix& A, const Hierarchical& like, bool copy) {
  assert(get_n_rows(A) == get_n_rows(like));
  assert(get_n_cols(A) == get_n_cols(like));
  return split_omm(
    A,
    IndexRange(0, get_n_rows(A)).split_like(like, ALONG_COL),
    IndexRange(0, get_n_cols(A)).split_like(like, ALONG_ROW),
    copy
  );
}

define_method(
  Hierarchical, split_omm,
  (
    const Dense& A,
    const std::vector<IndexRange>& row_splits,
    const std::vector<IndexRange>& col_splits,
    bool copy
  )
) {
  Hierarchical out(row_splits.size(), col_splits.size());
  std::vector<Dense> result = A.split(row_splits, col_splits, copy);
  for (int64_t i=0; i<out.dim[0]; ++i) {
    for (int64_t j=0; j<out.dim[1]; ++j) {
      out(i, j) = std::move(result[i*out.dim[1]+j]);
    }
  }
  return out;
}

define_method(
  Hierarchical, split_omm,
  (
    const LowRank& A,
    const std::vector<IndexRange>& row_splits,
    const std::vector<IndexRange>& col_splits,
    bool copy
  )
) {
  Hierarchical out(row_splits.size(), col_splits.size());
  Hierarchical U_splits;
  if (row_splits.size() > 1) {
    U_splits = split_omm(
      A.U, row_splits, {IndexRange(0, get_n_cols(A.U))}, copy
    );
  } else {
    U_splits = Hierarchical(1, 1);
    if (copy) {
      U_splits(0, 0) = Dense(A.U);
    } else {
      U_splits(0, 0) = shallow_copy(A.U);
    }
  }
  Hierarchical V_splits;
  if (col_splits.size() > 1) {
    V_splits = split_omm(
      A.V, {IndexRange(0, get_n_rows(A.V))}, col_splits, copy
    );
  } else {
    V_splits = Hierarchical(1, 1);
    if (copy) {
      V_splits(0, 0) = Dense(A.V);
    } else {
      V_splits(0, 0) = shallow_copy(A.V);
    }
  }
  for (uint64_t i=0; i<row_splits.size(); ++i) {
    for (uint64_t j=0; j<col_splits.size(); ++j) {
      out(i, j) = LowRank(U_splits[i], A.S, V_splits[j], copy);
    }
  }
  return out;
}

define_method(
  Hierarchical, split_omm,
  (
    const Hierarchical& A,
    const std::vector<IndexRange>& row_splits,
    const std::vector<IndexRange>& col_splits,
    bool copy
  )
) {
  if (
    (row_splits.size() != uint64_t(A.dim[0]))
    || (col_splits.size() != uint64_t(A.dim[1]))
  ) {
    std::abort();
  }
  Hierarchical out(row_splits.size(), col_splits.size());
  for (uint64_t i=0; i<row_splits.size(); ++i) {
    for (uint64_t j=0; j<col_splits.size(); ++j) {
      if (
        (row_splits[i].n != get_n_rows(A(i, j)))
        || (col_splits[j].n != get_n_cols(A(i, j)))
      ) std::abort();
      if (copy) {
        out(i, j) = A(i, j);
      } else {
        out(i, j) = shallow_copy(A(i, j));
      }
    }
  }
  return out;
}

define_method(
  Hierarchical, split_omm,
  (
    const Matrix& A,
    const std::vector<IndexRange>&, const std::vector<IndexRange>&, bool
  )
) {
  omm_error_handler("split", {A}, __FILE__, __LINE__);
  std::abort();
}

MatrixProxy shallow_copy(const Matrix& A) {
  return shallow_copy_omm(A);
}

define_method(MatrixProxy, shallow_copy_omm, (const Dense& A)) {
  // TODO Having this work for Dense might not be desirable
  return A.shallow_copy();
}

define_method(MatrixProxy, shallow_copy_omm, (const LowRank& A)) {
  LowRank scopy(A.U, A.S, A.V, false);
  scopy.eps = A.eps;
  return scopy;
}

define_method(MatrixProxy, shallow_copy_omm, (const Hierarchical& A)) {
  Hierarchical new_shallow_copy(A.dim[0], A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      new_shallow_copy(i, j) = shallow_copy(A(i, j));
    }
  }
  return new_shallow_copy;
}

define_method(MatrixProxy, shallow_copy_omm, (const Matrix& A)) {
  omm_error_handler("shallow_copy", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
