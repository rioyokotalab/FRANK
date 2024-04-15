#include "hicma/operations/misc.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/empty.h"
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
#include <iostream>


namespace hicma
{

// explicit template initialization (these are the only available types)
template Hierarchical<float> split(const Matrix&, int64_t, int64_t, bool);
template Hierarchical<double> split(const Matrix&, int64_t, int64_t, bool);
template Hierarchical<float> split(const Matrix&, const Hierarchical<float>&, bool);
template Hierarchical<double> split(const Matrix&, const Hierarchical<double>&, bool);
template double cond(Dense<float>);
template double cond(Dense<double>);
template double cond_inf(Dense<float>);
template double cond_inf(Dense<double>);

// TODO change return value to float?
template<typename T>
double cond(Dense<T> A) {
  double norm_A = norm(A);
  inverse(A);
  double norm_inv = norm(A);
  return std::sqrt(norm_A) * std::sqrt(norm_inv);
  //int64_t k = std::min(A.dim[0], A.dim[1]);
  //std::vector<T> S = get_singular_values(A);
  //std::cout<<S[0]<<" vs "<<S[1]<<" vs "<<S[k-3]<<" vs "<<S[k-2]<<" vs "<<S[k-1]<<std::endl;
  //return (S[0] / S[k-1]);
}

// TODO change return value to float?
template<typename T>
double cond_inf(Dense<T> A) {
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<T> S = get_singular_values(A);
  return (((double)S[0]) / S[k-1]);
}

// TODO this is not used anywhere, so no template for now
std::vector<double> equallySpacedVector(int64_t N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int64_t i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

template<typename T>
Hierarchical<T> split(
  const Matrix& A, int64_t n_row_blocks, int64_t n_col_blocks, bool copy
) {
  return split_omm(
    A,
    IndexRange(0, get_n_rows(A)).split(n_row_blocks),
    IndexRange(0, get_n_cols(A)).split(n_col_blocks),
    copy
  );
}

template<typename T>
Hierarchical<T> split(const Matrix& A, const Hierarchical<T>& like, bool copy) {
  assert(get_n_rows(A) == get_n_rows(like));
  assert(get_n_cols(A) == get_n_cols(like));
  return split_omm(
    A,
    IndexRange(0, get_n_rows(A)).split_like(like, ALONG_COL),
    IndexRange(0, get_n_cols(A)).split_like(like, ALONG_ROW),
    copy
  );
}

template<typename T>
MatrixProxy split_dense(const Dense<T>& A, const std::vector<IndexRange>& row_splits, const std::vector<IndexRange>& col_splits, bool copy) {
  Hierarchical<T> out(row_splits.size(), col_splits.size());
  std::vector<Dense<T>> result = A.split(row_splits, col_splits, copy);
  for (int64_t i=0; i<out.dim[0]; ++i) {
    for (int64_t j=0; j<out.dim[1]; ++j) {
      out(i, j) = std::move(result[i*out.dim[1]+j]);
    }
  }
  return std::move(out);
}

define_method(
  MatrixProxy, split_omm,
  (
    const Dense<float>& A,
    const std::vector<IndexRange>& row_splits,
    const std::vector<IndexRange>& col_splits,
    bool copy
  )
) {
  return split_dense(A, row_splits, col_splits, copy);
}

define_method(
  MatrixProxy, split_omm,
  (
    const Dense<double>& A,
    const std::vector<IndexRange>& row_splits,
    const std::vector<IndexRange>& col_splits,
    bool copy
  )
) {
  return split_dense(A, row_splits, col_splits, copy);
}

template<typename T>
MatrixProxy split_low_rank(const LowRank<T>& A, const std::vector<IndexRange>& row_splits, const std::vector<IndexRange>& col_splits, bool copy) {
  Hierarchical<T> out(row_splits.size(), col_splits.size());
  Hierarchical<T> U_splits;
  if (row_splits.size() > 1) {
    U_splits = split_omm(
      A.U, row_splits, {IndexRange(0, get_n_cols(A.U))}, copy
    );
  } else {
    U_splits = Hierarchical<T>(1, 1);
    if (copy) {
      U_splits(0, 0) = Dense<T>(A.U);
    } else {
      U_splits(0, 0) = shallow_copy(A.U);
    }
  }
  Hierarchical<T> V_splits;
  if (col_splits.size() > 1) {
    V_splits = split_omm(
      A.V, {IndexRange(0, get_n_rows(A.V))}, col_splits, copy
    );
  } else {
    V_splits = Hierarchical<T>(1, 1);
    if (copy) {
      V_splits(0, 0) = Dense<T>(A.V);
    } else {
      V_splits(0, 0) = shallow_copy(A.V);
    }
  }
  for (uint64_t i=0; i<row_splits.size(); ++i) {
    for (uint64_t j=0; j<col_splits.size(); ++j) {
      out(i, j) = LowRank<T>(U_splits[i], A.S, V_splits[j], copy);
    }
  }
  return std::move(out);
}

define_method(
  MatrixProxy, split_omm,
  (
    const LowRank<float>& A,
    const std::vector<IndexRange>& row_splits,
    const std::vector<IndexRange>& col_splits,
    bool copy
  )
) {
  return split_low_rank(A, row_splits, col_splits, copy);
}

define_method(
  MatrixProxy, split_omm,
  (
    const LowRank<double>& A,
    const std::vector<IndexRange>& row_splits,
    const std::vector<IndexRange>& col_splits,
    bool copy
  )
) {
  return split_low_rank(A, row_splits, col_splits, copy);
}

template<typename T>
MatrixProxy split_hierarchical(const Hierarchical<T>& A, const std::vector<IndexRange>& row_splits, const std::vector<IndexRange>& col_splits, bool copy) {
 if (
    (row_splits.size() != uint64_t(A.dim[0]))
    || (col_splits.size() != uint64_t(A.dim[1]))
  ) {
    std::abort();
  }
  Hierarchical<T> out(row_splits.size(), col_splits.size());
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
  return std::move(out);
}

define_method(
  MatrixProxy, split_omm,
  (
    const Hierarchical<float>& A,
    const std::vector<IndexRange>& row_splits,
    const std::vector<IndexRange>& col_splits,
    bool copy
  )
) {
  return split_hierarchical(A, row_splits, col_splits, copy);
}

define_method(
  MatrixProxy, split_omm,
  (
    const Hierarchical<double>& A,
    const std::vector<IndexRange>& row_splits,
    const std::vector<IndexRange>& col_splits,
    bool copy
  )
) {
  return split_hierarchical(A, row_splits, col_splits, copy);
}

define_method(
  MatrixProxy, split_omm,
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

define_method(MatrixProxy, shallow_copy_omm, (const Dense<float>& A)) {
  // TODO Having this work for Dense might not be desirable
  return A.shallow_copy();
}

define_method(MatrixProxy, shallow_copy_omm, (const Dense<double>& A)) {
  // TODO Having this work for Dense might not be desirable
  return A.shallow_copy();
}

template<typename T>
MatrixProxy hierarchical_shallow_copy(const Hierarchical<T>& A) {
  Hierarchical<T> new_shallow_copy(A.dim[0], A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      new_shallow_copy(i, j) = shallow_copy(A(i, j));
    }
  }
  return std::move(new_shallow_copy);
}

define_method(MatrixProxy, shallow_copy_omm, (const Hierarchical<float>& A)) {
  return hierarchical_shallow_copy(A);
}

define_method(MatrixProxy, shallow_copy_omm, (const Hierarchical<double>& A)) {
  return hierarchical_shallow_copy(A);
}

define_method(MatrixProxy, shallow_copy_omm, (const Matrix& A)) {
  omm_error_handler("shallow_copy", {A}, __FILE__, __LINE__);
  std::abort();
}

define_method(MatrixProxy, convert_omm, (const Hierarchical<double>& A, int64_t rank)) {
  return std::move(Hierarchical<float>(A, rank));
}

define_method(MatrixProxy, convert_omm, (const Hierarchical<float>& A, int64_t rank)) {
  return std::move(Hierarchical<double>(A, rank));
}

define_method(MatrixProxy, convert_omm, (const LowRank<double>& A, int64_t rank)) {
  Dense<float> U(A.dim[0], rank);
  A.U.copy_cut(U, A.dim[0], rank);
  Dense<float> S(rank, rank);
  A.S.copy_cut(S, rank, rank);
  Dense<float> V(rank, A.dim[1]);
  A.V.copy_cut(V, rank, A.dim[1]);
  LowRank<float> LR(std::move(U), std::move(S), std::move(V));
  return std::move(LR);;
}

define_method(MatrixProxy, convert_omm, (const LowRank<float>& A, int64_t rank)) {
  Dense<double> U(A.dim[0], rank);
  A.U.copy_cut(U, A.dim[0], rank);
  Dense<double> S(rank, rank);
  A.S.copy_cut(S, rank, rank);
  Dense<double> V(rank, A.dim[1]);
  A.V.copy_cut(V, rank, A.dim[1]);
  LowRank<double> LR(std::move(U), std::move(S), std::move(V));
  return std::move(LR);
}

define_method(MatrixProxy, convert_omm, (const Dense<double>& A, int64_t)) {
  return std::move(Dense<float>(A));
}

define_method(MatrixProxy, convert_omm, (const Dense<float>& A, int64_t)) {
  return std::move(Dense<double>(A));
}

define_method(MatrixProxy, convert_omm, (const Empty& A, int64_t)) {
  return std::move(Empty());
}

define_method(MatrixProxy, convert_omm, (const Matrix& A, int64_t)) {
  omm_error_handler("convert_omm", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
