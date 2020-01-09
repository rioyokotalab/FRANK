#ifndef hicma_functions_h
#define hicma_functions_h

#include <vector>

namespace hicma {

class Dense;

void zeros(
  Dense& A,
  std::vector<double>& x
);

void identity(
  Dense& A,
  std::vector<double>& x
);

void random_normal(
  Dense& A,
  std::vector<double>& x
);

void random_uniform(
  Dense& A,
  std::vector<double>& x
);

void arange(
  Dense& A,
  std::vector<double>& x
);

void laplace1d(
  Dense& A,
  std::vector<double>& x
);

void cauchy2d(
  std::vector<double>& data,
  std::vector<std::vector<double>>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
);

void laplacend(
  std::vector<double>& data,
  std::vector<std::vector<double>>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
);

void helmholtznd(
  std::vector<double>& data,
  std::vector<std::vector<double>>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin
);

bool is_admissible_nd(
  std::vector<std::vector<double>>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin,
  const double& admis
);

bool is_admissible_nd_morton(
  std::vector<std::vector<double>>& x,
  const int& ni,
  const int& nj,
  const int& i_begin,
  const int& j_begin,
  const double& admis
);

} // namespace hicma

#endif // hicma_functions_h
