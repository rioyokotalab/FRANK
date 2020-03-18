#ifndef operations_misc_h
#define operations_misc_h

#include <vector>


namespace hicma
{
  class Dense;

  double cond(Dense A);

  double diam(std::vector<double>& x, const int& n, const int& offset);

  double mean(std::vector<double>& x, const int& n, const int& offset);

  std::vector<int> getIndex(int dim, int mortonIndex);

  int getMortonIndex(std::vector<int> index, int level);

  std::vector<double> equallySpacedVector(int N, double minVal, double maxVal);

  void getSubmatrix(const Dense& A, int ni, int nj, int i_begin, int j_begin, Dense& out);

} // namespace hicma

#endif // operations_misc_h
