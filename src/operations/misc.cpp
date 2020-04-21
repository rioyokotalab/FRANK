#include "hicma/operations/misc.h"

#include "hicma/node.h"
#include "hicma/dense.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <utility>
#include <cassert>

#include "yorel/multi_methods.hpp"

namespace hicma
{

  double cond(Dense A) {
    int k = std::min(A.dim[0], A.dim[1]);
    Dense S(k);
    A.svd(S);
    return (S(0, 0) / S(k-1, 0));
  }

  double diam(std::vector<double>& x, const int& n, const int& offset) {
    double xmax = *std::max_element(x.begin()+offset, x.begin()+offset+n);
    double xmin = *std::min_element(x.begin()+offset, x.begin()+offset+n);
    return std::abs(xmax-xmin);
  }

  double mean(std::vector<double>& x, const int& n, const int& offset) {
    return std::accumulate(x.begin()+offset, x.begin()+offset+n, 0.0)/n;
  }

  std::vector<int> getIndex(int dim, int mortonIndex) {
    std::vector<int> index(dim, 0);
    int d = 0, level = 0;
    while (mortonIndex != 0) {
      index[d] += (mortonIndex % 2) * (1 << level);
      mortonIndex >>= 1;
      d = (d + 1) % dim;
      if (d == 0) level++;
    }
    return index;
  }

  int getMortonIndex(std::vector<int> index, int level) {
    int mortonIndex = 0;
    for(int lev=0; lev<level; lev++) {
      for(int d=0; d<(int)index.size(); d++) {
        mortonIndex += index[d] % 2 << (index.size() * lev + d);
        index[d] >>= 1;
      }
    }
    return mortonIndex;
  }

  std::vector<double> equallySpacedVector(int N, double minVal, double maxVal) {
    std::vector<double> res(N, 0.0);
    double rnge = maxVal - minVal;
    for(int i=0; i<N; i++) {
      res[i] = minVal + ((double)i/(double)rnge);
    }
    return res;
  }

  void getSubmatrix(const Dense& A, int ni, int nj, int i_begin, int j_begin, Dense& out) {
    assert(out.dim[0] == ni);
    assert(out.dim[1] == nj);
    for(int i=0; i<ni; i++)
      for(int j=0; j<nj; j++) {
        out(i, j) = A(i+i_begin, j+j_begin);
      }
  }

} // namespace hicma
