#ifndef hicma_operations_misc_h
#define hicma_operations_misc_h

#include <cstdint>
#include <vector>


namespace hicma
{

class Node;
class Dense;

int64_t get_n_rows(const Node&);

int64_t get_n_cols(const Node&);

double cond(Dense A);

double diam(const std::vector<double>& x, int64_t n, int64_t offset);

double mean(const std::vector<double>& x, int64_t n, int64_t offset);

std::vector<int64_t> getIndex(int64_t dim, int64_t mortonIndex);

int64_t getMortonIndex(std::vector<int64_t> index, int64_t level);

std::vector<double> equallySpacedVector(
  int64_t N, double minVal, double maxVal);

// TODO This is the same as Dense.get_part()
void getSubmatrix(
  const Dense& A,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start,
  Dense& out
);

double norm(const Node&);

void transpose(Node&);

} // namespace hicma

#endif // hicma_operations_misc_h