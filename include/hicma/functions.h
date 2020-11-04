#ifndef hicma_functions_h
#define hicma_functions_h

#include <cstdint>
#include <vector>


namespace hicma
{

void zeros(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void identity(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void random_normal(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void random_uniform(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void arange(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void cauchy2d(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void laplacend(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void helmholtznd(
  double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

bool is_admissible_nd(
  const std::vector<std::vector<double>>& x,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start,
  double admis
);

bool is_admissible_nd_morton(
  const std::vector<std::vector<double>>& x,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start,
  double admis
);

 namespace starsh {
   
 }

} // namespace hicma

#endif // hicma_functions_h
