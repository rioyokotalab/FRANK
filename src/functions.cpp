#include "hicma/functions.h"

#include <cmath>
#include <random>
#include <cassert>

namespace hicma
{

// explicit template instantiation (this is a pain TODO make better (abstract class?))
template void zeros(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void zeros(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void identity(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void identity(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void random_normal(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void random_normal(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void random_uniform(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void random_uniform(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void arange(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void arange(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void cauchy2d(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void cauchy2d(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void laplacend(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void laplacend(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void helmholtznd(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void helmholtznd(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

template<typename T>
void zeros(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>&, int64_t, int64_t
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = 0;
    }
  }
}

template<typename T>
void identity(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>&, int64_t row_start, int64_t col_start
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = row_start+i == col_start+j ? 1 : 0;
    }
  }
}

template<typename T>
void random_normal(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>&, int64_t, int64_t
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      // relies on implicit type conversion
      A[i*A_stride+j] = dist(gen);
    }
  }
}

template<typename T>
void random_uniform(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>&, int64_t, int64_t
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      // relies on implicit type conversion
      A[i*A_stride+j] = dist(gen);
    }
  }
}

template<typename T>
void arange(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>&, int64_t row_start, int64_t col_start
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      // relies on implicit type conversion
      A[i*A_stride+j] = ((row_start+i)*A_cols+col_start+j);
    }
  }
}

template<typename T>
void cauchy2d(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) {
  assert(x.size()>1);
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      // double sgn = (arc4random() % 2 ? 1.0 : -1.0);
      double rij = (x[0][i+row_start] - x[1][j+col_start]) + 1e-2;
      // relies on implicit type conversion
      A[i*A_stride+j] = 1.0 / rij;
    }
  }
}

template<typename T>
void laplacend(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
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
      // relies on implicit type conversion
      A[i*A_stride+j] = 1 / (std::sqrt(rij) + 1e-3);
    }
  }
}

template<typename T>
void helmholtznd(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
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
      // relies on implicit type conversion
      A[i*A_stride+j] = std::exp(-1.0 * rij) / (std::sqrt(rij) + 1e-3);
    }
  }
}

} // namespace hicma
