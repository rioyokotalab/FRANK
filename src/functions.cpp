#include "hicma/functions.h"

#include "hicma/operations/misc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// #include <starsh.h>
// #include <starsh-randtlr.h>
// #include <starsh-electrodynamics.h>
// #include <starsh-spatial.h>

namespace hicma
{

void zeros(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>&, int64_t, int64_t
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = 0;
    }
  }
}

void identity(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>&, int64_t row_start, int64_t col_start
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = row_start+i == col_start+j ? 1 : 0;
    }
  }
}

void random_normal(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>&, int64_t, int64_t
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  std::normal_distribution<float> dist(0.0, 1.0);
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = dist(gen);
    }
  }
}

void random_uniform(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>&, int64_t, int64_t
) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = dist(gen);
    }
  }
}

void arange(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>&, int64_t, int64_t
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      A[i*A_stride+j] = (float)(i*A_cols+j);
    }
  }
}

void cauchy2d(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      // float sgn = (arc4random() % 2 ? 1.0 : -1.0);
      float rij = (x[0][i+row_start] - x[1][j+col_start]) + 1e-2;
      A[i*A_stride+j] = 1.0 / rij;
    }
  }
}

void laplacend(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      float rij = 0.0;
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
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
) {
  for (uint64_t i=0; i<A_rows; i++) {
    for (uint64_t j=0; j<A_cols; j++) {
      float rij = 0.0;
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

bool is_admissible_nd(
  const std::vector<std::vector<float>>& x,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start,
  float admis
) {
  std::vector<float> diamsI, diamsJ, centerI, centerJ;
  for(size_t k=0; k<x.size(); k++) {
    diamsI.push_back(diam(x[k], n_rows, row_start));
    diamsJ.push_back(diam(x[k], n_cols, col_start));
    centerI.push_back(mean(x[k], n_rows, row_start));
    centerJ.push_back(mean(x[k], n_cols, col_start));
  }
  float diamI = *std::max_element(diamsI.begin(), diamsI.end());
  float diamJ = *std::max_element(diamsJ.begin(), diamsJ.end());
  float dist = 0.0;
  for(size_t k=0; k<x.size(); k++) {
    dist += (centerI[k]-centerJ[k])*(centerI[k]-centerJ[k]);
  }
  dist = std::sqrt(dist);
  return (std::max(diamI, diamJ) <= (admis * dist));
}

bool is_admissible_nd_morton(
  const std::vector<std::vector<float>>& x,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start,
  float admis
) {
  std::vector<float> diamsI, diamsJ, centerI, centerJ;
  for(size_t k=0; k<x.size(); k++) {
    diamsI.push_back(diam(x[k], n_rows, row_start));
    diamsJ.push_back(diam(x[k], n_cols, col_start));
    centerI.push_back(mean(x[k], n_rows, row_start));
    centerJ.push_back(mean(x[k], n_cols, col_start));
  }
  float diamI = *std::max_element(diamsI.begin(), diamsI.end());
  float diamJ = *std::max_element(diamsJ.begin(), diamsJ.end());
  //Compute distance based on morton index of box
  int64_t boxSize = std::min(n_rows, n_cols);
  int64_t npartitions = x[0].size()/boxSize;
  int64_t level = (int64_t)log2((float)npartitions);
  std::vector<int64_t> indexI(x.size(), 0), indexJ(x.size(), 0);
  for(size_t k=0; k<x.size(); k++) {
    indexI[k] = row_start/boxSize;
    indexJ[k] = col_start/boxSize;
  }
  float dist = std::abs(
    getMortonIndex(indexI, level)
    - getMortonIndex(indexJ, level)
  );
  return (std::max(diamI, diamJ) <= (admis * dist));
}

  // namespace starsh {
  //   STARSH_kernel *kernel;
  //   void *starsh_data;
  //   std::vector<STARSH_int> starsh_index;

  //   void exp_kernel_prepare(
  //     int64_t N, float beta, float nu, float noise,float sigma, int ndim
  //   ) {
  //     enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;
  //     if (ndim == 2) {
  //       kernel = starsh_ssdata_block_exp_kernel_2d;
  //     } else if (ndim == 3) {
  //       kernel = starsh_ssdata_block_exp_kernel_3d;
  //     }
  //     starsh_ssdata_generate(
  //       (STARSH_ssdata **)&starsh_data, N, ndim, beta, nu, noise, place, sigma
  //     );
  //     for (int j = 0; j < N; ++j) {
  //       starsh_index.push_back(j);
  //     }
  //   }

  //   void exp_kernel_fill(
  //     float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  //     const std::vector<std::vector<float>>&,
  //     int64_t row_start, int64_t col_start
  //   ) {
  //     kernel(
  //       A_rows, A_cols,
  //       starsh_index.data() + row_start, starsh_index.data() + col_start,
  //       starsh_data, starsh_data,
  //       A, A_stride
  //     );
  //   }

  //   void exp_kernel_cleanup() {
  //     starsh_ssdata_free((STARSH_ssdata *)starsh_data);
  //   }
  // }

} // namespace hicma
