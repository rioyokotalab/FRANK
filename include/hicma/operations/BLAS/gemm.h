#ifndef hicma_operations_BLAS_gemm_h
#define hicma_operations_BLAS_gemm_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;
class Dense;

void gemm(const Node&, const Node&, Node&, const double, const double);

void gemm(
  const Dense& A, const Dense& B, Dense& C,
  const bool TransA, const bool TransB,
  const double& alpha, const double& beta
);

MULTI_METHOD(
  gemm_omm, void,
  const virtual_<Node>&,
  const virtual_<Node>&,
  virtual_<Node>&,
  const double,
  const double
);

} // namespace hicma

#endif // hicma_operations_BLAS_gemm_h