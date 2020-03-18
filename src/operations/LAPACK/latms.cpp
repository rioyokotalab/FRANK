#include "hicma/operations/LAPACK/latms.h"

#include "hicma/classes/dense.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <cassert>
#include <vector>


namespace hicma
{

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
  ) {
    LAPACKE_dlatms(
      LAPACK_ROW_MAJOR, A.dim[0], A.dim[1],
      dist, &iseed[0], sym, &d[0], mode, cond, dmax, kl, ku, pack,
      &A, A.stride
    );
  }

} // namespace hicma
