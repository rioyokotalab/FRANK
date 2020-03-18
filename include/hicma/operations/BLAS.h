#ifndef hicma_operations_BLAS_h
#define hicma_operations_BLAS_h


namespace hicma
{

class Node;

void gemm(const Node&, const Node&, Node&, double, double);

void gemm(
  const Node& A, const Node& B, Node& C,
  bool TransA, bool TransB,
  double alpha, double beta
);

void trmm(
  const Node& A, Node& B,
  const char& side, const char& uplo, const char& trans, const char& diag,
  const double& alpha
);

void trmm(
  const Node& A, Node& B,
  const char& side, const char& uplo,
  const double& alpha
);

// TODO consider splitting left and right trsm into separate functions! That
// would allow nicer syntax: ltrsm(Tri, A, 'u/l') and rtrsm(A, Tri, 'u/l').
void trsm(const Node&, Node&, const char& uplo, bool left=true);

} // namespace hicma

#endif // hicma_operations_BLAS_h
