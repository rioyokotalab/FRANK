#ifndef hicma_operations_LAPACK_h
#define hicma_operations_LAPACK_h

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class Matrix;
class MatrixProxy;
class Dense;

std::tuple<Dense, std::vector<int64_t>> geqp3(Matrix& A);

void geqrt(Matrix&, Matrix&);

void geqrt2(Dense&, Dense&);

std::tuple<MatrixProxy, MatrixProxy> getrf(Matrix&);

std::tuple<Dense, std::vector<int64_t>> one_sided_id(Matrix& A, int64_t k);

// TODO Does this need to be in the header?
Dense get_cols(const Dense& A, std::vector<int64_t> P);

std::tuple<Dense, Dense, Dense> id(Matrix& A, int64_t k);

void larfb(const Matrix&, const Matrix&, Matrix&, bool);

void latms(
  const char& dist,
  std::vector<int>& iseed,
  const char& sym,
  std::vector<double>& d,
  int mode,
  double cond,
  double dmax,
  int kl, int ku,
  const char& pack,
  Dense& A
);

void qr(Matrix&, Matrix&, Matrix&);

// TODO Does this need to be in the header?
bool need_split(const Matrix&);

// TODO Does this need to be in the header?
std::tuple<Dense, Dense> make_left_orthogonal(const Matrix&);

// TODO Does this need to be in the header?
void update_splitted_size(const Matrix&, int64_t&, int64_t&);

// TODO Does this need to be in the header?
MatrixProxy split_by_column(const Matrix&, Matrix&, int64_t&);

// TODO Does this need to be in the header?
MatrixProxy concat_columns(
  const Matrix&, const Matrix&, const Matrix&, int64_t&);

void zero_lowtri(Matrix&);

void zero_whole(Matrix&);

void rq(Matrix&, Matrix&, Matrix&);

std::tuple<Dense, Dense, Dense> svd(Dense& A);

std::tuple<Dense, Dense, Dense> sdd(Dense& A);

// TODO Does this need to be in the header?
Dense get_singular_values(Dense& A);

void tpmqrt(const Matrix&, const Matrix&, Matrix&, Matrix&, bool);

void tpqrt(Matrix&, Matrix&, Matrix&);

} // namespace hicma

#endif // hicma_operations_LAPACK_h
