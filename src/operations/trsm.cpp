#include "hicma/operations/trsm.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/gemm.h"
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

void trsm(const Node& A, Node& B, const char& uplo) {
  trsm_omm(A, B, uplo);
}

BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Hierarchical& A, Hierarchical& B,
  const char& uplo
) {
  if (B.dim[1] == 1) {
    switch (uplo) {
    case 'l' :
      for (int i=0; i<B.dim[0]; i++) {
        for (int j=0; j<i; j++) {
          gemm(A(i,j), B[j], B[i], -1, 1);
        }
        trsm(A(i,i), B[i], 'l');
      }
      break;
    case 'u' :
      for (int i=B.dim[0]-1; i>=0; i--) {
        for (int j=B.dim[0]-1; j>i; j--) {
          gemm(A(i,j), B[j], B[i], -1, 1);
        }
        trsm(A(i,i), B[i], 'u');
      }
      break;
    default :
      std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
      abort();
    }
  }
  else {
    switch (uplo) {
    case 'l' :
      for (int j=0; j<B.dim[1]; j++) {
        for (int i=0; i<B.dim[0]; i++) {
          for (int k=0; k<i; k++) {
            gemm(A(i,k), B(k,j), B(i,j), -1, 1);
          }
          trsm(A(i,i), B(i,j), 'l');
        }
      }
      break;
    case 'u' :
      for (int i=0; i<B.dim[0]; i++) {
        for (int j=0; j<B.dim[1]; j++) {
          for (int k=0; k<j; k++) {
            gemm(B(i,k), A(k,j), B(i,j), -1, 1);
          }
          trsm(A(j,j), B(i,j), 'u');
        }
      }
      break;
    default :
      std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
      abort();
    }
  }
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Dense& A, Dense& B,
  const char& uplo
) {
  start("-DTRSM");
  if (B.dim[1] == 1) {
    switch (uplo) {
    case 'l' :
      cblas_dtrsm(
        CblasRowMajor,
        CblasLeft, CblasLower,
        CblasNoTrans, CblasUnit,
        B.dim[0], B.dim[1],
        1,
        &A[0], A.dim[1],
        &B.data[0], B.dim[1]
      );
      break;
    case 'u' :
      cblas_dtrsm(
        CblasRowMajor,
        CblasLeft, CblasUpper,
        CblasNoTrans, CblasNonUnit,
        B.dim[0], B.dim[1],
        1,
        &A[0], A.dim[1],
        &B.data[0], B.dim[1]
      );
      break;
    default :
      std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
      abort();
    }
  }
  else {
    switch (uplo) {
    case 'l' :
      cblas_dtrsm(
        CblasRowMajor,
        CblasLeft, CblasLower,
        CblasNoTrans, CblasUnit,
        B.dim[0], B.dim[1],
        1,
        &A[0], A.dim[1],
        &B.data[0], B.dim[1]
      );
      break;
    case 'u' :
      cblas_dtrsm(
        CblasRowMajor,
        CblasRight, CblasUpper,
        CblasNoTrans, CblasNonUnit,
        B.dim[0], B.dim[1],
        1,
        &A[0], A.dim[1],
        &B.data[0], B.dim[1]
      );
      break;
    default :
      std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
      abort();
    }
  }
  stop("-DTRSM",false);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Dense& A, LowRank& B,
  const char& uplo
) {
  switch (uplo) {
  case 'l' :
    trsm(A, B.U, uplo);
    break;
  case 'u' :
    trsm(A, B.V, uplo);
    break;
  }
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Hierarchical& A, LowRank& B,
  const char& uplo
) {
  switch (uplo) {
  case 'l' :
    {
      Hierarchical UH(B.U, A.dim[0], 1);
      trsm(A, UH, uplo);
      B.U = Dense(UH);
      break;
    }
  case 'u' :
    {
      Hierarchical VH(B.V, 1, A.dim[1]);
      trsm(A, VH, uplo);
      B.V = Dense(VH);
      break;
    }
  }
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Hierarchical& A, Dense& B,
  const char& uplo
) {
  switch (uplo) {
  case 'l' :
    {
      Hierarchical BH(B, A.dim[0], 1);
      trsm(A, BH, uplo);
      B = Dense(BH);
      break;
    }
  case 'u' :
    {
      Hierarchical BH(B, 1, A.dim[1]);
      trsm(A, BH, uplo);
      B = Dense(BH);
      break;
    }
  default :
    std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
    abort();
  }
} END_SPECIALIZATION;

// Fallback default, abort with error message
BEGIN_SPECIALIZATION(
  trsm_omm, void,
  const Node& A, Node& B,
  const char& uplo
) {
  std::cerr << "trsm(";
  std::cerr << A.type() << "," << B.type();
  std::cerr << ") undefined." << std::endl;
  abort();
} END_SPECIALIZATION;

} // namespace hicma
