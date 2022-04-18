#include "hicma/functions.h"

#include "hicma/operations/misc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace hicma
{

void zeros(
  double* A, const uint64_t A_rows, const uint64_t A_cols, const uint64_t A_stride,
  const std::vector<std::vector<double>>&, const int64_t, const int64_t
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = 0;
    }
  }
}

void identity(
  double* A, const uint64_t A_rows, const uint64_t A_cols, const uint64_t A_stride,
  const std::vector<std::vector<double>>&, const int64_t row_start, const int64_t col_start
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = row_start+i == col_start+j ? 1 : 0;
    }
  }
}

void random_normal(
  double* A, const uint64_t A_rows, const uint64_t A_cols, const uint64_t A_stride,
  const std::vector<std::vector<double>>&, const int64_t, const int64_t
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = dist(gen);
    }
  }
}

void random_uniform(
  double* A, const uint64_t A_rows, const uint64_t A_cols, const uint64_t A_stride,
  const std::vector<std::vector<double>>&, const int64_t, const int64_t
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = dist(gen);
    }
  }
}

void arange(
  double* A, const uint64_t A_rows, const uint64_t A_cols, const uint64_t A_stride,
  const std::vector<std::vector<double>>&, const int64_t, const int64_t
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = (double)(i*A_cols+j);
    }
  }
}

void cauchy2d(
  double* A, const uint64_t A_rows, const uint64_t A_cols, const uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  const int64_t row_start, const int64_t col_start
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      const double rij = (x[0][i+row_start] - x[1][j+col_start]) + 1e-2;
      A[i*A_stride+j] = 1.0 / rij;
    }
  }
}

void laplacend(
  double* A, const uint64_t A_rows, const uint64_t A_cols, const uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  const int64_t row_start, const int64_t col_start
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      double rij = 0.0;
      for(size_t k=0; k<x.size(); k++) {
        rij += (
          (x[k][i+row_start] - x[k][j+col_start])
          * (x[k][i+row_start] - x[k][j+col_start])
        );
      }
      A[i*A_stride+j] = 1 / (std::sqrt(rij) + 1e-3);
    }
  }
}

void helmholtznd(
  double* A, const uint64_t A_rows, const uint64_t A_cols, const uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  const int64_t row_start, const int64_t col_start
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      double rij = 0.0;
      for(size_t k=0; k<x.size(); k++) {
        rij += (
          (x[k][i+row_start] - x[k][j+col_start])
          * (x[k][i+row_start] - x[k][j+col_start])
        );
      }
      A[i*A_stride+j] = std::exp(-1.0 * rij) / (std::sqrt(rij) + 1e-3);
    }
  }
}

} // namespace hicma
