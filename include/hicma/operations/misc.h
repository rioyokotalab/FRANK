#ifndef operations_misc_h
#define operations_misc_h

#include <vector>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{
  class Dense;

  std::vector<int> getIndex(int dim, int mortonIndex);

  int getMortonIndex(std::vector<int> index, int level);

  std::vector<double> equallySpacedVector(int N, double minVal, double maxVal);

  double cond(Dense A);
} // namespace hicma

#endif // operations_misc_h
