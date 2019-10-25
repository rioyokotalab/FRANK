#ifndef operations_gemm_h
#define operations_gemm_h

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif
#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;
class Dense;

void gemm(const Node&, const Node&, Node&, const double, const double);

void gemm(
  const Dense& A, const Dense& B, Dense& C,
  const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
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

#endif // operations_gemm_h
