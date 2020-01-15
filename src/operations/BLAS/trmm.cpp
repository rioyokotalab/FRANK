#include "hicma/operations/BLAS/trmm.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/util/timer.h"

#include <cassert>
#include <iostream>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

  void trmm(
    const Node& A, Node& B,
    const char& side, const char& uplo, const char& trans, const char& diag,
    const double& alpha
  ) {
    trmm_omm(A, B, side, uplo, trans, diag, alpha);
  }

  void trmm(
    const Node& A, Node& B,
    const char& side, const char& uplo,
    const double& alpha
  ) {
    trmm_omm(A, B, side, uplo, 'n', 'n', alpha);
  }

  BEGIN_SPECIALIZATION(
    trmm_omm, void,
    const Dense& A, Dense& B,
    const char& side, const char& uplo, const char& trans, const char& diag,
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
      B.dim[0], B.dim[1], alpha, &A, A.stride, &B, B.stride
    );
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
    trmm_omm, void,
    const Dense& A, LowRank& B,
    const char& side, const char& uplo,  const char& trans, const char& diag,
    const double& alpha
  ) {
    assert(A.dim[0] == A.dim[1]);
    assert(A.dim[0] == (side == 'l' ? B.dim[0] : B.dim[1]));
    if(side == 'l')
      trmm(A, B.U(), side, uplo, trans, diag, alpha);
    else if(side == 'r')
      trmm(A, B.V(), side, uplo, trans, diag, alpha);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
    trmm_omm, void,
    const Hierarchical& A, Hierarchical& B,
    const char& side, const char& uplo, const char& trans, const char& diag,
    const double& alpha
  ) {
    assert(A.dim[0] == A.dim[1]);
    assert(A.dim[0] == (side == 'l' ? B.dim[0] : B.dim[1]));
    assert(uplo == 'u'); //TODO implement for lower triangular
    Hierarchical B_copy(B);
    if(uplo == 'u') {
      if(side == 'l') {
        for(int i=0; i<A.dim[0]; i++) {
          for(int j=0; j<B.dim[1]; j++) {
            for(int k=i; k<A.dim[1]; k++) {
              if(k == i) {
                trmm(A(i, k), B(k, j), side, uplo, trans, diag, alpha);
              }
              else {
                gemm(A(i, k), B_copy(k, j), B(i, j), alpha, 1);
              }
            }
          }
        }
      }
      else if(side == 'r') {
        for(int i=0; i<B.dim[0]; i++) {
          for(int j=0; j<A.dim[1]; j++) {
            for(int k=j; k>=0; k--) {
              if(k == j) {
                trmm(A(k, j), B(i, k), side, uplo, trans, diag, alpha);
              }
              else {
                gemm(B_copy(i, k), A(k, j), B(i, j), alpha, 1);
              }
            }
          }
        }
      }
    }
  } END_SPECIALIZATION;

  // Fallback default, abort with error message
  BEGIN_SPECIALIZATION(
    trmm_omm, void,
    const Node& A, Node& B,
    const char& side, const char& uplo, const char& trans, const char& diag,
    const double& alpha
  ) {
    std::cerr << "trmm(";
    std::cerr << A.type() << "," << B.type();
    std::cerr << ") undefined." << std::endl;
    abort();
  } END_SPECIALIZATION;

} // namespace hicma
