#include "hicma/operations/trmm.h"

#include "hicma/node.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/util/timer.h"

#include <cassert>
#include <iostream>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

  void trmm(const Node& A, Node& B, const char& side, const char& uplo, const char& trans, const char& diag, const double& alpha) {
    trmm_omm(A, B, side, uplo, trans, diag, alpha);
  }

  BEGIN_SPECIALIZATION(
                       trmm_omm, void,
                       const Dense& A, Dense& B,
                       const char& side,
                       const char& uplo,
                       const char& trans,
                       const char& diag,
                       const double& alpha
                       ) {
    assert(A.dim[0] == A.dim[1]);
    assert(A.dim[0] == (side == 'l' ? B.dim[0] : B.dim[1]));
    cblas_dtrmm(
                CblasRowMajor,
                side == 'l' ? CblasLeft : CblasRight,
                uplo == 'u' ? CblasUpper : CblasLower,
                trans == 't' ? CblasTrans : CblasNoTrans,
                diag == 'u' ? CblasUnit : CblasNonUnit,
                B.dim[0], B.dim[1], alpha, &A[0], A.dim[1], &B[0], B.dim[1]);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
                       trmm_omm, void,
                       const Dense& A, LowRank& B,
                       const char& side,
                       const char& uplo,
                       const char& trans,
                       const char& diag,
                       const double& alpha
                       ) {
    assert(A.dim[0] == A.dim[1]);
    assert(A.dim[0] == (side == 'l' ? B.dim[0] : B.dim[1]));
    if(side == 'l')
      trmm(A, B.U, side, uplo, trans, diag, alpha);
    else if(side == 'r')
      trmm(A, B.V, side, uplo, trans, diag, alpha);
  } END_SPECIALIZATION;

  // Fallback default, abort with error message
  BEGIN_SPECIALIZATION(
                       trmm_omm, void,
                       const Node& A, Node& B,
                       const char& side,
                       const char& uplo,
                       const char& trans,
                       const char& diag,
                       const double& alpha
                       ) {
    std::cerr << "trmm(";
    std::cerr << A.type() << "," << B.type();
    std::cerr << ") undefined." << std::endl;
    abort();
  } END_SPECIALIZATION;

} // namespace hicma
