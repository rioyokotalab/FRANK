#ifndef hicma_operations_BLAS_gemm_h
#define hicma_operations_BLAS_gemm_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma
{

void gemm(const Node&, const Node&, Node&, double, double);

void gemm(
  const Node& A, const Node& B, Node& C,
  bool TransA, bool TransB,
  double alpha, double beta
);

declare_method(
  void, gemm_omm,
  (
    virtual_<const Node&>, virtual_<const Node&>, virtual_<Node&>,
    double, double
  )
);

} // namespace hicma

#endif // hicma_operations_BLAS_gemm_h
