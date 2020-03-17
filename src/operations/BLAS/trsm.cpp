#include "hicma/operations/BLAS/trsm.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/no_copy_split.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/util/timer.h"

#include <iostream>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif
#include "yorel/multi_methods.hpp"

namespace hicma
{

void trsm(const Node& A, Node& B, const char& uplo, bool left) {
  trsm_omm(A, B, uplo, left);
}

BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Hierarchical& A, Hierarchical& B,
  const char& uplo, bool left
) {
  switch (uplo) {
  case 'l' :
    if (left) {
      for (int j=0; j<B.dim[1]; j++) {
        for (int i=0; i<B.dim[0]; i++) {
          for (int k=0; k<i; k++) {
            gemm(A(i,k), B(k,j), B(i,j), -1, 1);
          }
          trsm(A(i,i), B(i,j), 'l', left);
        }
      }
    } else {
      std::cerr << " Right lower trsm not implemented yet!" << std::endl;
      abort();
    }
    break;
  case 'u' :
    if (left) {
      if (B.dim[1] == 1) {
        for (int i=B.dim[0]-1; i>=0; i--) {
          for (int j=B.dim[0]-1; j>i; j--) {
            gemm(A(i,j), B[j], B[i], -1, 1);
          }
          trsm(A(i,i), B[i], 'u', left);
        }
      } else {
        std::cerr << "Hierarchical left upper trsm not implemented yet!" << std::endl;
        abort();
      }
    } else {
      for (int i=0; i<B.dim[0]; i++) {
        for (int j=0; j<B.dim[1]; j++) {
          for (int k=0; k<j; k++) {
            gemm(B(i,k), A(k,j), B(i,j), -1, 1);
          }
          trsm(A(j,j), B(i,j), 'u', left);
        }
      }
    }
    break;
  default :
    std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
    abort();
  }
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Dense& A, Dense& B,
  const char& uplo, bool left
) {
  timing::start("DTRSM");
  switch (uplo) {
  case 'l' :
    cblas_dtrsm(
      CblasRowMajor,
      left?CblasLeft:CblasRight, CblasLower,
      CblasNoTrans, CblasUnit,
      B.dim[0], B.dim[1],
      1,
      &A, A.stride,
      &B, B.stride
    );
    break;
  case 'u' :
    cblas_dtrsm(
      CblasRowMajor,
      left?CblasLeft:CblasRight, CblasUpper,
      CblasNoTrans, CblasNonUnit,
      B.dim[0], B.dim[1],
      1,
      &A, A.stride,
      &B, B.stride
    );
    break;
  default :
    std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
    abort();
  }
  timing::stop("DTRSM");
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Node& A, LowRank& B,
  const char& uplo, bool left
) {
  switch (uplo) {
  case 'l' :
    trsm(A, B.U(), uplo, left);
    break;
  case 'u' :
    trsm(A, B.V(), uplo, left);
    break;
  default :
    std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
    abort();
  }
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Hierarchical& A, Dense& B,
  const char& uplo, bool left
) {
  NoCopySplit BH(B, left?A.dim[0]:1, left?1:A.dim[1]);
  switch (uplo) {
  case 'l' :
    trsm(A, BH, uplo, left);
    break;
  case 'u' :
    trsm(A, BH, uplo, left);
    break;
  default :
    std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
    abort();
  }
} END_SPECIALIZATION;

// Fallback default, abort with error message
BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Node& A, Node& B,
  [[maybe_unused]] const char& uplo, [[maybe_unused]] bool left
) {
  std::cerr << "trsm(";
  std::cerr << A.type() << "," << B.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
