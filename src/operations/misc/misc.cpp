#include "hicma/operations/misc.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/util/omm_error_handler.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
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

MatrixProxy get_part(
  const Matrix& A,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start,
  bool copy
) {
  return get_part_omm(A, n_rows, n_cols, row_start, col_start, copy);
}

MatrixProxy get_part(
  const Matrix& A,
  const ClusterTree& node,
  bool copy
) {
  return get_part_omm(
    A, node.rows.n, node.cols.n, node.rows.start, node.cols.start, copy);
}

define_method(
  MatrixProxy, get_part_omm,
  (
    const Dense& A,
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    bool copy
  )
) {
  return Dense(A, n_rows, n_cols, row_start, col_start, copy);
}

define_method(
  MatrixProxy, get_part_omm,
  (
    const SharedBasis& A,
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    bool copy
  )
) {
  return get_part(A.transfer_mat(), n_rows, n_cols, row_start, col_start, copy);
}

define_method(
  MatrixProxy, get_part_omm,
  (
    const LowRank& A,
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    bool copy
  )
) {
  return LowRank(A, n_rows, n_cols, row_start, col_start, copy);
}

define_method(
  MatrixProxy, get_part_omm,
  (
    const Matrix& A,
    [[maybe_unused]] int64_t n_rows, [[maybe_unused]] int64_t n_cols,
    [[maybe_unused]] int64_t row_start, [[maybe_unused]] int64_t col_start,
    [[maybe_unused]] bool copy
  )
) {
  omm_error_handler("get_part", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
