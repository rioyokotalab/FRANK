#include "FRANK/operations/LAPACK.h"

#include "FRANK/classes/dense.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <vector>


namespace FRANK
{

  void latms(
    const char& dist,
    std::vector<int>& iseed,
    const char& sym,
    std::vector<double>& d,
    const int mode,
    const double cond,
    const double dmax,
    const int kl, const int ku,
    const char& pack,
    Dense& A
  ) {
    LAPACKE_dlatms(
      LAPACK_ROW_MAJOR, A.dim[0], A.dim[1],
      dist, &iseed[0], sym, &d[0], mode, cond, dmax, kl, ku, pack,
      &A, A.stride
    );
  }

} // namespace FRANK
