#include "hicma/operations/latms.h"

#include "hicma/dense.h"
#include <vector>
#include <cassert>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

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
             ) {
    LAPACKE_dlatms(LAPACK_ROW_MAJOR, m, n, dist, &iseed[0], sym, &d[0], mode, cond, dmax, kl, ku, pack, &A[0], A.dim[1]);
  }

} // namespace hicma
