#include "hicma/operations.h"

#include "hicma/any.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/util/timer.h"
#include "hicma/util/print.h"

#include "yorel/multi_methods.hpp"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

namespace hicma
{

void trsm(const Any& A, Any& B, const char& uplo) {
  trsm(*A.ptr.get(), *B.ptr.get(), uplo);
}
void trsm(const Any& A, Node& B, const char& uplo) {
  trsm(*A.ptr.get(), B, uplo);
}
void trsm(const Node& A, Any& B, const char& uplo) {
  trsm(A, *B.ptr.get(), uplo);
}

void trsm(const Node& A, Node& B, const char& uplo) {
  trsm_omm(A, B, uplo);
}

BEGIN_SPECIALIZATION(trsm_omm, void, const Hierarchical& A, Hierarchical& B, const char& uplo) {
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
          gemm_row(A, B, B, i, j, 0, i, -1, 1);
          trsm(A(i,i), B(i,j), 'l');
        }
      }
      break;
    case 'u' :
      for (int i=0; i<B.dim[0]; i++) {
        for (int j=0; j<B.dim[1]; j++) {
          gemm_row(B, A, B, i, j, 0, j, -1, 1);
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

BEGIN_SPECIALIZATION(trsm_omm, void, const Dense& A, Dense& B, const char& uplo) {
  start("-DTRSM");
  if (B.dim[1] == 1) {
    switch (uplo) {
    case 'l' :
      cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                  B.dim[0], B.dim[1], 1, &A[0], A.dim[1], &B.data[0], B.dim[1]);
      break;
    case 'u' :
      cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                  B.dim[0], B.dim[1], 1, &A[0], A.dim[1], &B.data[0], B.dim[1]);
      break;
    default :
      std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
      abort();
    }
  }
  else {
    switch (uplo) {
    case 'l' :
      cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                  B.dim[0], B.dim[1], 1, &A[0], A.dim[1], &B.data[0], B.dim[1]);
      break;
    case 'u' :
      cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                  B.dim[0], B.dim[1], 1, &A[0], A.dim[1], &B.data[0], B.dim[1]);
      break;
    default :
      std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
      abort();
    }
  }
  stop("-DTRSM",false);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(trsm_omm, void, const Dense& A, LowRank& B, const char& uplo) {
  switch (uplo) {
  case 'l' :
    trsm(A, B.U, uplo);
    break;
  case 'u' :
    trsm(A, B.V, uplo);
    break;
  }
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(trsm_omm, void, const Hierarchical& A, LowRank& B, const char& uplo) {
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

BEGIN_SPECIALIZATION(trsm_omm, void, const Hierarchical& A, Dense& B, const char& uplo) {
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
BEGIN_SPECIALIZATION(trsm_omm, void, const Node& A, Node& B, const char& uplo) {
  print_undefined(__func__, A.type(), B.type());
  abort();
} END_SPECIALIZATION;

} // namespace hicma