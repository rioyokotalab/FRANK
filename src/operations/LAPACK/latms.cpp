#include "hicma/operations/LAPACK.h"

#include "hicma/classes/dense.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <vector>


namespace hicma
{

  void latms(
    const char& dist,
    std::vector<int>& iseed,
    const char& sym,
    std::vector<float>& d,
    int mode,
    float cond,
    float dmax,
    int kl, int ku,
    const char& pack,
    Dense& A
  ) {
    LAPACKE_slatms(
      LAPACK_ROW_MAJOR, A.dim[0], A.dim[1],
      dist, &iseed[0], sym, &d[0], mode, cond, dmax, kl, ku, pack,
      &A, A.stride
    );
  }

} // namespace hicma
