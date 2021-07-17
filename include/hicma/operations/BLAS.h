#ifndef hicma_operations_BLAS_h
#define hicma_operations_BLAS_h

#include "hicma/classes/dense.h"


namespace hicma
{

class Matrix;

void gemm(
  const Matrix& A, const Matrix& B, Matrix& C,
  float alpha=1, float beta=1,
  bool TransA=false, bool TransB=false
);

MatrixProxy gemm(
  const Matrix& A, const Matrix& B,
  float alpha=1,
  bool TransA=false, bool TransB=false
);

enum { TRSM_UPPER, TRSM_LOWER };
enum { TRSM_LEFT, TRSM_RIGHT };

void trmm(
  const Matrix& A, Matrix& B,
  const char& side, const char& uplo, const char& trans, const char& diag,
  float alpha
);

void trmm(
  const Matrix& A, Matrix& B,
  const char& side, const char& uplo,
  float alpha
);

void trsm(const Matrix&, Matrix&, int uplo, int lr=TRSM_LEFT);

} // namespace hicma

#endif // hicma_operations_BLAS_h
