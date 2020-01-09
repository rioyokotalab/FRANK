#ifndef operations_latms_h
#define operations_latms_h

#include <vector>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

  class Node;
  class Dense;

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

} // namespace hicma

#endif
