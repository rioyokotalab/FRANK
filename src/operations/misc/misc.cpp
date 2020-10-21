#include "hicma/operations/misc.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
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

double cond(Dense A) {
  int64_t k = std::min(A.dim[0], A.dim[1]);
  Dense S = get_singular_values(A);
  return (S[0] / S[k-1]);
}

double diam(const std::vector<double>& x, int64_t n, int64_t offset) {
  double xmax = *std::max_element(x.begin()+offset, x.begin()+offset+n);
  double xmin = *std::min_element(x.begin()+offset, x.begin()+offset+n);
  return std::abs(xmax-xmin);
}

double mean(const std::vector<double>& x, int64_t n, int64_t offset) {
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

std::vector<double> equallySpacedVector(int64_t N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int64_t i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
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

declare_method(
  LowRank, resolve_nested_basis,
  (virtual_<const Matrix&>, virtual_<const Matrix&>, const Dense&, bool)
)

define_method(
  LowRank, resolve_nested_basis,
  (const NestedBasis& U, const NestedBasis& V, const Dense& S, bool)
) {
  return LowRank(
    U.sub_bases, gemm(gemm(U.translation, S), V.translation), V.sub_bases,
    false
  );
}

define_method(
  LowRank, resolve_nested_basis,
  (const Hierarchical& U, const NestedBasis& V, const Dense& S, bool)
) {
  return LowRank(U, gemm(S, V.translation), V.sub_bases, false);
}

define_method(
  LowRank, resolve_nested_basis,
  (const NestedBasis& U, const Hierarchical& V, const Dense& S, bool)
) {
  return LowRank(U.sub_bases, gemm(U.translation, S), V, false);
}

define_method(
  LowRank, resolve_nested_basis,
  (const Dense& U, const Dense& V, const Dense& S, bool copy_S)
) {
  return LowRank(U, S, V, copy_S);
}

define_method(
  LowRank, resolve_nested_basis,
  (const Matrix& U, const Matrix& V, const Dense&, bool)
) {
  omm_error_handler("resolve_nested_basis", {U, V}, __FILE__, __LINE__);
  std::abort();
}

declare_method(
  NestedBasis, resolve_nested_basis,
  (virtual_<const Matrix&>, const Dense&, bool, bool)
)

define_method(
  NestedBasis, resolve_nested_basis,
  (const Dense& basis, const Dense& S, bool, bool is_col_basis)
) {
  // TODO No difference for copying/non-copying version atm!
  return NestedBasis(basis.share(), S.share(), is_col_basis);
}

define_method(
  NestedBasis, resolve_nested_basis,
  (const NestedBasis& basis, const Dense& S, bool, bool is_col_basis)
) {
  // TODO No difference for copying/non-copying version atm!
  return NestedBasis(
    basis.sub_bases,
    is_col_basis ? gemm(basis.translation, S) : gemm(S, basis.translation),
    is_col_basis
  );
}

define_method(
  NestedBasis, resolve_nested_basis,
  (const Matrix& basis, const Dense&, bool, bool)
) {
  omm_error_handler("resolve_nested_basis", {basis}, __FILE__, __LINE__);
  std::abort();
}

define_method(
  Hierarchical, split_omm,
  (
    const NestedBasis& A,
    const std::vector<IndexRange>& row_splits,
    const std::vector<IndexRange>& col_splits,
    bool copy
  )
) {
  assert(
    (A.is_col_basis() && col_splits.size() == 1)
    || (A.is_row_basis() && row_splits.size() == 1)
  );
  Hierarchical out(row_splits.size(), col_splits.size());
  // TODO Possibly wrong dimensions for subbasis!
  Hierarchical sub_basis_split = split_omm(
    A.sub_bases, row_splits, col_splits, copy
  );
  for (int64_t i=0; i<out.dim[A.is_col_basis() ? 0 : 1]; ++i) {
    out[i] = resolve_nested_basis(
      sub_basis_split[i], A.translation, copy, A.is_col_basis()
    );
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
      U_splits(0, 0) = A.U;
    } else {
      U_splits(0, 0) = share_basis(A.U);
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
      V_splits(0, 0) = A.V;
    } else {
      V_splits(0, 0) = share_basis(A.V);
    }
  }
  for (uint64_t i=0; i<row_splits.size(); ++i) {
    for (uint64_t j=0; j<col_splits.size(); ++j) {
      out(i, j) = resolve_nested_basis(U_splits[i], V_splits[j], A.S, copy);
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
  // TODO Consider ways to remove warning generated by line below
  if ((row_splits.size() != A.dim[0]) || (col_splits.size() != A.dim[1])) {
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
        out(i, j) = share_basis(A(i, j));
      }
    }
  }
  return out;
}

define_method(
  Hierarchical, split_omm,
  (
    const Matrix& A,
    [[maybe_unused]] const std::vector<IndexRange>&,
    [[maybe_unused]] const std::vector<IndexRange>&,
    [[maybe_unused]] bool)
) {
  omm_error_handler("split", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
