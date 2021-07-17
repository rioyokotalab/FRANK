#include "hicma/operations/misc.h"
#include "hicma/extension_headers/classes.h"

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

float cond(Dense A) {
  int64_t k = std::min(A.dim[0], A.dim[1]);
  std::vector<float> S = get_singular_values(A);
  return (S[0] / S[k-1]);
}

float diam(const std::vector<float>& x, int64_t n, int64_t offset) {
  float xmax = *std::max_element(x.begin()+offset, x.begin()+offset+n);
  float xmin = *std::min_element(x.begin()+offset, x.begin()+offset+n);
  return std::abs(xmax-xmin);
}

float mean(const std::vector<float>& x, int64_t n, int64_t offset) {
  return std::accumulate(x.begin()+offset, x.begin()+offset+n, 0.0)/n;
}

std::vector<int64_t> getIndex(int64_t dim, int64_t mortonIndex) {
  std::vector<int64_t> index(dim, 0);
  int64_t d = 0, level = 0;
  while (mortonIndex != 0) {
    index[d] += (mortonIndex % 2) * (1 << level);
    mortonIndex >>= 1;
    d = (d + 1) % dim;
    if (d == 0) level++;
  }
  return index;
}

int64_t getMortonIndex(std::vector<int64_t> index, int64_t level) {
  int64_t mortonIndex = 0;
  for(int64_t lev=0; lev<level; lev++) {
    for(size_t d=0; d<index.size(); d++) {
      mortonIndex += index[d] % 2 << (index.size() * lev + d);
      index[d] >>= 1;
    }
  }
  return mortonIndex;
}

std::vector<float> equallySpacedVector(int64_t N, float minVal, float maxVal) {
  std::vector<float> res(N, 0.0);
  float rnge = maxVal - minVal;
  for(int64_t i=0; i<N; i++) {
    res[i] = minVal + ((float)i/(float)rnge);
  }
  return res;
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

define_method(MatrixProxy, shallow_copy_omm, (const Hierarchical& A)) {
  Hierarchical new_shallow_copy(A.dim[0], A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      new_shallow_copy(i, j) = shallow_copy(A(i, j));
    }
  }
  return std::move(new_shallow_copy);
}

define_method(MatrixProxy, shallow_copy_omm, (const Matrix& A)) {
  omm_error_handler("shallow_copy", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
