#ifndef hicma_functions_h
#define hicma_functions_h

#include <cstdint>
#include <vector>


namespace hicma
{

class Dense;

void zeros(
  Dense& A, const std::vector<double>& x, int64_t row_start, int64_t col_start);

void identity(
  Dense& A, const std::vector<double>& x, int64_t row_start, int64_t col_start);

void random_normal(
  Dense& A, const std::vector<double>& x, int64_t row_start, int64_t col_start);

void random_uniform(
  Dense& A, const std::vector<double>& x, int64_t row_start, int64_t col_start);

void arange(
  Dense& A, const std::vector<double>& x, int64_t row_start, int64_t col_start);

void laplace1d(
  Dense& A, const std::vector<double>& x, int64_t row_start, int64_t col_start);

void cauchy2d(
  Dense& A,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void laplacend(
  Dense& A,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

void helmholtznd(
  Dense& A,
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

} // namespace hicma

#endif // hicma_functions_h
