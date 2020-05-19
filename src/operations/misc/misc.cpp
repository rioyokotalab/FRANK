#include "hicma/operations/misc.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/operations/LAPACK.h"

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

void getSubmatrix(
  const Dense& A,
  int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
  Dense& out
) {
  assert(out.dim[0] == n_rows);
  assert(out.dim[1] == n_cols);
  for(int64_t i=0; i<n_rows; i++)
    for(int64_t j=0; j<n_cols; j++) {
      out(i, j) = A(i+row_start, j+col_start);
    }
}

} // namespace hicma
