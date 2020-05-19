#ifndef hicma_operations_BLAS_h
#define hicma_operations_BLAS_h

#include "hicma/classes/dense.h"


namespace hicma
{

class Matrix;

void gemm(const Matrix&, const Matrix&, Matrix&, double, double);

void gemm(
  const Matrix& A, const Matrix& B, Matrix& C,
  bool TransA, bool TransB,
  double alpha, double beta
);

Dense gemm(
  const Matrix& A, const Matrix& B,
  double alpha=1,
  bool TransA=false, bool TransB=false
);

enum { TRSM_UPPER, TRSM_LOWER };
enum { TRSM_LEFT, TRSM_RIGHT };

void trmm(
  const Matrix& A, Matrix& B,
  const char& side, const char& uplo, const char& trans, const char& diag,
  double alpha
);

void trmm(
  const Matrix& A, Matrix& B,
  const char& side, const char& uplo,
  double alpha
);

void trsm(const Matrix&, Matrix&, int uplo, int lr=TRSM_LEFT);

} // namespace hicma

#endif // hicma_operations_BLAS_h
