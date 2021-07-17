#ifndef hicma_functions_h
#define hicma_functions_h

#include <cstdint>
#include <vector>

namespace hicma
{
void zeros(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
);

void identity(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
);

void random_normal(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
);

void random_uniform(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
);

void arange(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
);

void cauchy2d(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
);

void laplacend(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
);

void helmholtznd(
  float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<float>>& x,
  int64_t row_start, int64_t col_start
);

bool is_admissible_nd(
  const std::vector<std::vector<float>>& x,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start,
  float admis
);

bool is_admissible_nd_morton(
  const std::vector<std::vector<float>>& x,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start,
  float admis
);

  // namespace starsh {
  //   void exp_kernel_prepare(int64_t N, float beta, float nu, float noise,
  //                           float sigma, int ndim);
  //   void exp_kernel_fill(float* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  //                        const std::vector<std::vector<float>>& x,
  //                        int64_t row_start, int64_t col_start);
  //   void exp_kernel_cleanup();
  // }

} // namespace hicma

#endif // hicma_functions_h
