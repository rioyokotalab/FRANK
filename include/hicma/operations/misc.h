#ifndef hicma_operations_misc_h
#define hicma_operations_misc_h

#include "hicma/classes/hierarchical.h"
#include "hicma/classes/matrix_proxy.h"

#include <cstdint>
#include <vector>


namespace hicma
{

class ClusterTree;
class Dense;
class Hierarchical;
class LowRank;
class Matrix;

int64_t get_n_rows(const Matrix&);

int64_t get_n_cols(const Matrix&);

double cond(Dense A);

double diam(const std::vector<double>& x, int64_t n, int64_t offset);

double mean(const std::vector<double>& x, int64_t n, int64_t offset);

std::vector<int64_t> getIndex(int64_t dim, int64_t mortonIndex);

int64_t getMortonIndex(std::vector<int64_t> index, int64_t level);

std::vector<double> equallySpacedVector(
  int64_t N, double minVal, double maxVal);

Hierarchical split(
  const Matrix& A, int64_t n_row_blocks, int64_t n_col_blocks, bool copy=false
);

Hierarchical split(const Matrix& A, const Hierarchical& like, bool copy=false);

double norm(const Matrix&);

void recompress_col(Matrix& AU, const Matrix& BU, Dense& AS, const Dense& BS);

void recompress_row(Matrix& AV, const Matrix& BV, Dense& AS, const Dense& BS);

LowRank recombine_col(Hierarchical& A, MatrixProxy& V);

LowRank recombine_row(Hierarchical& A, MatrixProxy& U);

MatrixProxy transpose(const Matrix&);

MatrixProxy resize(const Matrix&, int64_t n_rows, int64_t n_cols);

} // namespace hicma

#endif // hicma_operations_misc_h
