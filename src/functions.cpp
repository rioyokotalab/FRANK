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
template void gaussiannd(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void gaussiannd(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void imqnd(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void imqnd(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void maternnd(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);
template void maternnd(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
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
  size_t d = x.size() -1;
  assert(x[d].size() == 2);
  T l = x[d][0];
  T shift = x[d][1];

  double min = 1;

  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      double rij = 0.0;
      for(size_t k=0; k<d; k++) {
        rij += (
          (x[k][i+row_start] - x[k][j+col_start])
          * (x[k][i+row_start] - x[k][j+col_start])
        );
      }
      // relies on implicit type conversion
      if (i == j) 
        A[i*A_stride+j] = 0;
      else {
        double dist = std::sqrt(rij);
        if (min > dist)
          min = dist;
        A[i*A_stride+j] = 1 / (l * dist);
        //A[i*A_stride+j] = 1 / (std::sqrt(rij) + 1e-3);
      }
    }
  }
  for (uint64_t i=0; i<A_rows; i++) {
    A[i*A_stride+i] = 1 / (l * min) + shift;
  }
}

template<typename T>
void helmholtznd(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) {
  size_t d = x.size() -1;
  assert(x[d].size() == 2);
  T l = x[d][0];
  T shift = x[d][1];

  double min = 1;

  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      double rij = 0.0;
      for(size_t k=0; k<d; k++) {
        rij += (
          (x[k][i+row_start] - x[k][j+col_start])
          * (x[k][i+row_start] - x[k][j+col_start])
        );
      }
      if (i == j) 
        A[i*A_stride+j] = 1;
      else {
        double dist = std::sqrt(rij);
        if (min > dist)
          min = dist;
        A[i*A_stride+j] = std::exp(-l * rij) / (dist);
        // relies on implicit type conversion
        //A[i*A_stride+j] = std::exp(-1.0 * rij) / (std::sqrt(rij) + shift);
      }
    }
  }
  for (uint64_t i=0; i<A_rows; i++) {
    A[i*A_stride+i] = (1 / min) + shift;
  }
}

// we use the 3rd entry of x to pass the parameter l
template<typename T>
void gaussiannd(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) {
  size_t d = x.size() -1;
  assert(x[d].size() == 2);
  T l = x[d][0];
  T shift = x[d][1];

  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      double rij = 0.0;
      for(size_t k=0; k<d; k++) {
        rij += (
          (x[k][i+row_start] - x[k][j+col_start])
          * (x[k][i+row_start] - x[k][j+col_start])
        );
      }
      // relies on implicit type conversion
      A[i*A_stride+j] = std::exp(-l * std::sqrt(rij));
      // diagonal shift
      if (i == j) {
        A[i*A_stride+j] += shift;
      }
    }
  }
}

// we use the 3rd entry of x to pass the parameter l
template<typename T>
void imqnd(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) {
  size_t d = x.size() -1;
  assert(x[d].size() == 2);
  T l = x[d][0];
  T shift = x[d][1];

  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      double rij = 0.0;
      for(size_t k=0; k<d; k++) {
        rij += (
          (x[k][i+row_start] - x[k][j+col_start])
          * (x[k][i+row_start] - x[k][j+col_start])
        );
      }
      // relies on implicit type conversion
      A[i*A_stride+j] = 1.0 / std::sqrt(1.0 + l * std::sqrt(rij));
      // diagonal shift
      if (i == j) {
        A[i*A_stride+j] += shift;
      }
    }
  }
}

// we use the 3rd entry of x to pass the parameter l
template<typename T>
void maternnd(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
) {
    size_t d = x.size() -1;
  assert(x[d].size() == 2);
  T l = x[d][0];
  T shift = x[d][1];

  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      double rij = 0.0;
      for(size_t k=0; k<d; k++) {
        rij += (
          (x[k][i+row_start] - x[k][j+col_start])
          * (x[k][i+row_start] - x[k][j+col_start])
        );
      }
      rij = std::sqrt(rij);
      // relies on implicit type conversion
      A[i*A_stride+j] = 1.0 + std::sqrt(3) * l * rij
                        * std::exp(-std::sqrt(3)* l * rij);
      // diagonal shift
      if (i == j) {
        A[i*A_stride+j] += shift;
      }
    }
  }
}

} // namespace hicma
