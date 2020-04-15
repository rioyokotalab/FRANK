#ifndef hicma_operations_BLAS_h
#define hicma_operations_BLAS_h

#include "hicma/classes/dense.h"


namespace hicma
{

class Node;

void gemm(const Node&, const Node&, Node&, double, double);

void gemm(
  const Node& A, const Node& B, Node& C,
  bool TransA, bool TransB,
  double alpha, double beta
);

Dense gemm(
  const Node& A, const Node& B,
  double alpha=1,
  bool TransA=false, bool TransB=false
);

enum { TRSM_UPPER, TRSM_LOWER };
enum { TRSM_LEFT, TRSM_RIGHT };

void trmm(
  const Node& A, Node& B,
  const char& side, const char& uplo, const char& trans, const char& diag,
  double alpha
);

void trmm(
  const Node& A, Node& B,
  const char& side, const char& uplo,
  double alpha
);

void trsm(const Node&, Node&, int uplo, int lr=TRSM_LEFT);

} // namespace hicma

#endif // hicma_operations_BLAS_h
