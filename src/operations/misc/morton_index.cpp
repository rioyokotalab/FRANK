#include "hicma/operations/misc.h"

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

std::vector<int64_t> getCartesianIndex(const int64_t dim, int64_t mortonIndex) {
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

int64_t getMortonIndex(std::vector<int64_t>& iX, const int64_t level) {
  int64_t mortonIndex = 0;
  for(int64_t lev=0; lev<level; lev++) {
    for(size_t d=0; d<iX.size(); d++) {
      mortonIndex += (iX[d] & 1) << (iX.size() * lev + d);
      iX[d] >>= 1;
    }
  }
  return mortonIndex;
}

int64_t getBoxNumber(const std::vector<double>& p, const int64_t level) {
  const int64_t nx = (1 << level);
  std::vector<int64_t> iX;
  for(size_t dim=0; dim<p.size(); dim++) {
    iX.push_back(p[dim] * nx);
  }
  return getMortonIndex(iX, level);
}

std::vector<std::vector<double>> normalizeWithinBox(std::vector<std::vector<double>>& x, const double eps) {
  std::vector<double> bmin;
  double bsize = 0.0;
  for(size_t dim=0; dim<x.size(); dim++) {
    const double xmin = *std::min_element(x[dim].begin(), x[dim].end());
    const double xmax = *std::max_element(x[dim].begin(), x[dim].end());
    bsize = std::max(bsize, xmax-xmin);
    bmin.push_back(xmin - eps);
  }
  bsize += 2*eps;
  std::vector<std::vector<double>> normalized(x);
  for(size_t dim=0; dim<x.size(); dim++) {
    for(size_t i=0; i<x[dim].size(); i++) {
      normalized[dim][i] = (x[dim][i] - bmin[dim]) / bsize;
    }
  }
  return normalized;
}

void sortByMortonIndex(std::vector<std::vector<double>> &x, const int64_t level, std::vector<int64_t>& perm) {
  std::vector<std::vector<double>> xn = normalizeWithinBox(x, 5e-1);
  std::vector<std::vector<double>> xnT (xn[0].size(), std::vector<double>());
  for(size_t i=0; i<xnT.size(); i++) {
    for(size_t dim=0; dim<xn.size(); dim++) {
      xnT[i].push_back(xn[dim][i]);
    }
  }
  std::vector<std::pair<int64_t, int64_t>> indexWithBoxNum;
  for(size_t i=0; i<xnT.size(); i++)
    indexWithBoxNum.push_back(std::make_pair((int64_t)i, getBoxNumber(xnT[i], level)));
  std::sort(indexWithBoxNum.begin(),
            indexWithBoxNum.end(),
            [](std::pair<int64_t, int64_t> a, std::pair<int64_t, int64_t> b) {
              return a.second < b.second;
            });
  std::vector<std::vector<double>> x_copy(x);
  for(size_t i=0; i<indexWithBoxNum.size(); i++) {
    perm[i] = indexWithBoxNum[i].first;
    for(size_t dim=0; dim<x.size(); dim++) {
      x[dim][i] = x_copy[dim][indexWithBoxNum[i].first];
    }
  }
}

} // namespace hicma
