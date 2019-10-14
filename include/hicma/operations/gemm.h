#ifndef operations_gemm_h
#define operations_gemm_h

#include "hicma/node_proxy.h"
#include "hicma/node.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif
#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

void gemm(const NodeProxy&, const NodeProxy&, NodeProxy&, const double, const double);
void gemm(const NodeProxy&, const NodeProxy&, Node&, const double, const double);
void gemm(const NodeProxy&, const Node&, NodeProxy&, const double, const double);
void gemm(const NodeProxy&, const Node&, Node&, const double, const double);
void gemm(const Node&, const NodeProxy&, NodeProxy&, const double, const double);
void gemm(const Node&, const NodeProxy&, Node&, const double, const double);
void gemm(const Node&, const Node&, NodeProxy&, const double, const double);

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

void gemm_row(
              const Hierarchical& A, const Hierarchical& B, Hierarchical& C,
              const int& i, const int& j, const int& k_min, const int& k_max,
              const double& alpha, const double& beta);

} // namespace hicma

#endif // operations_gemm_h
