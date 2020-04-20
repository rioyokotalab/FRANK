#include "hicma/functions.h"

#include "hicma/classes/dense.h"
#include "hicma/operations/misc/misc.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <iostream>


namespace hicma
{

void zeros(
  Dense& A, [[maybe_unused]] const std::vector<double>& x,
  [[maybe_unused]] int64_t row_start, [[maybe_unused]] int64_t col_start
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = 0;
    }
  }
}

void identity(
  Dense& A, [[maybe_unused]] const std::vector<double>& x,
  int64_t row_start, int64_t col_start
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = row_start+i == col_start+j ? 1 : 0;
    }
  }
}

void random_normal(
  Dense& A, [[maybe_unused]] const std::vector<double>& x,
  [[maybe_unused]] int64_t row_start, [[maybe_unused]] int64_t col_start
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = dist(gen);
    }
  }
}

void random_uniform(
  Dense& A, [[maybe_unused]] const std::vector<double>& x,
  [[maybe_unused]] int64_t row_start, [[maybe_unused]] int64_t col_start
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = dist(gen);
    }
  }
}

void arange(
  Dense& A, [[maybe_unused]] const std::vector<double>& x,
  [[maybe_unused]] int64_t row_start, [[maybe_unused]] int64_t col_start
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = (double)(i*A.dim[1]+j);
    }
  }
}

void laplace1d(
  Dense& A, const std::vector<double>& x, int64_t row_start, int64_t col_start
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = 1 / (
        std::abs(x[i+row_start] - x[j+col_start]) + 1e-3);
    }
  }
}

void cauchy2d(
  Dense& A,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      // double sgn = (arc4random() % 2 ? 1.0 : -1.0);
      double rij = (x[0][i+row_start] - x[1][j+col_start]) + 1e-2;
      A(i, j) = 1.0 / rij;
    }
  }
}

void laplacend(
  Dense& A,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      double rij = 0.0;
      for(size_t k=0; k<x.size(); k++) {
        rij += (
          (x[k][i+row_start] - x[k][j+col_start])
          * (x[k][i+row_start] - x[k][j+col_start])
        );
      }
      A(i, j) = 1 / (std::sqrt(rij) + 1e-3);
    }
  }
}

void helmholtznd(
  Dense& A,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      double rij = 0.0;
      for(size_t k=0; k<x.size(); k++) {
        rij += (
          (x[k][i+row_start] - x[k][j+col_start])
          * (x[k][i+row_start] - x[k][j+col_start])
        );
      }
      A(i, j) = std::exp(-1.0 * rij) / (std::sqrt(rij) + 1e-3);
    }
  }
}

bool is_admissible_nd(
  const std::vector<std::vector<double>>& x,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start,
  double admis
) {
  std::vector<double> diamsI, diamsJ, centerI, centerJ;
  for(size_t k=0; k<x.size(); k++) {
    diamsI.push_back(diam(x[k], n_rows, row_start));
    diamsJ.push_back(diam(x[k], n_cols, col_start));
    centerI.push_back(mean(x[k], n_rows, row_start));
    centerJ.push_back(mean(x[k], n_cols, col_start));
  }
  double diamI = *std::max_element(diamsI.begin(), diamsI.end());
  double diamJ = *std::max_element(diamsJ.begin(), diamsJ.end());
  double dist = 0.0;
  for(size_t k=0; k<x.size(); k++) {
    dist += (centerI[k]-centerJ[k])*(centerI[k]-centerJ[k]);
  }
  dist = std::sqrt(dist);
  return (std::max(diamI, diamJ) <= (admis * dist));
}

bool is_admissible_nd_morton(
  const std::vector<std::vector<double>>& x,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start,
  double admis
) {
  std::vector<double> diamsI, diamsJ, centerI, centerJ;
  for(size_t k=0; k<x.size(); k++) {
    diamsI.push_back(diam(x[k], n_rows, row_start));
    diamsJ.push_back(diam(x[k], n_cols, col_start));
    centerI.push_back(mean(x[k], n_rows, row_start));
    centerJ.push_back(mean(x[k], n_cols, col_start));
  }
  double diamI = *std::max_element(diamsI.begin(), diamsI.end());
  double diamJ = *std::max_element(diamsJ.begin(), diamsJ.end());
  //Compute distance based on morton index of box
  int64_t boxSize = std::min(n_rows, n_cols);
  int64_t npartitions = x[0].size()/boxSize;
  int64_t level = (int64_t)log2((double)npartitions);
  std::vector<int64_t> indexI(x.size(), 0), indexJ(x.size(), 0);
  for(size_t k=0; k<x.size(); k++) {
    indexI[k] = row_start/boxSize;
    indexJ[k] = col_start/boxSize;
  }
  double dist = std::abs(
    getMortonIndex(indexI, level)
    - getMortonIndex(indexJ, level)
  );
  return (std::max(diamI, diamJ) <= (admis * dist));
}


} // namespace hicma
