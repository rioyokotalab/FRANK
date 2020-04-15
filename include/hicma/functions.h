#ifndef hicma_functions_h
#define hicma_functions_h

#include <cstdint>
#include <vector>


namespace hicma
{

class Dense;

void zeros(
  Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin);

void identity(
  Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin);

void random_normal(
  Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin);

void random_uniform(
  Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin);

void arange(Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin);

void laplace1d(
  Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin);

void cauchy2d(
  std::vector<double>& data,
  std::vector<std::vector<double>>& x,
  int64_t ni, int64_t nj,
  int64_t i_begin, int64_t j_begin
);

void laplacend(
  std::vector<double>& data,
  std::vector<std::vector<double>>& x,
  int64_t ni, int64_t nj,
  int64_t i_begin, int64_t j_begin
);

void helmholtznd(
  std::vector<double>& data,
  std::vector<std::vector<double>>& x,
  int64_t ni, int64_t nj,
  int64_t i_begin, int64_t j_begin
);

bool is_admissible_nd(
  std::vector<std::vector<double>>& x,
  int64_t ni, int64_t nj,
  int64_t i_begin, int64_t j_begin,
  double admis
);

bool is_admissible_nd_morton(
  std::vector<std::vector<double>>& x,
  int64_t ni, int64_t nj,
  int64_t i_begin, int64_t j_begin,
  double admis
);

} // namespace hicma

#endif // hicma_functions_h
