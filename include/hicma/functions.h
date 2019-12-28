#ifndef hicma_functions_h
#define hicma_functions_h

#include <vector>

namespace hicma {

void zeros(
  std::vector<double>& data,
  std::vector<double>& x,
  int ni, int nj,
  int i_begin, int j_begin
);

void identity(
  std::vector<double>& data,
  std::vector<double>& x,
  int ni, int nj,
  int i_begin, int j_begin
);

void random_normal(
  std::vector<double>& data,
  std::vector<double>& x,
  int ni, int nj,
  int i_begin, int j_begin
);

void random_uniform(
  std::vector<double>& data,
  std::vector<double>& x,
  int ni, int nj,
  int i_begin, int j_begin
);

void latms(
  std::vector<double>& data,
  std::vector<double>& x,
  int ni, int nj,
  int i_begin, int j_begin
);

void arange(
  std::vector<double>& data,
  std::vector<double>& x,
  int ni, int nj,
  int i_begin, int j_begin
);

void laplace1d(
  std::vector<double>& data,
  std::vector<double>& x,
  int ni, int nj,
  int i_begin, int j_begin
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
