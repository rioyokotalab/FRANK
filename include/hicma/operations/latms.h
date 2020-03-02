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
             const int& m,
             const int& n,
             const char& dist,
             std::vector<int>& iseed,
             const char& sym,
             std::vector<double>& d,
             const int& mode,
             const double& cond,
             const double& dmax,
             const int& kl,
             const int& ku,
             const char& pack,
             Dense& A
             );

} // namespace hicma

#endif
