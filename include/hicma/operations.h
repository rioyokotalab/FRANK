#ifndef operations_h
#define operations_h

#include "any.h"
#include "node.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif
#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

void getrf(Any&);

void getrf(Node&);

MULTI_METHOD(
  getrf_omm, void,
  virtual_<Node>&
);


void trsm(const Any&, Any&, const char& uplo);
void trsm(const Any&, Node&, const char& uplo);
void trsm(const Node&, Any&, const char& uplo);

void trsm(const Node&, Node&, const char& uplo);

MULTI_METHOD(
  trsm_omm, void,
  const virtual_<Node>&,
  virtual_<Node>&,
  const char& uplo
);


void gemm(const Any&, const Any&, Any&, const double, const double);
void gemm(const Any&, const Any&, Node&, const double, const double);
void gemm(const Any&, const Node&, Any&, const double, const double);
void gemm(const Any&, const Node&, Node&, const double, const double);
void gemm(const Node&, const Any&, Any&, const double, const double);
void gemm(const Node&, const Any&, Node&, const double, const double);
void gemm(const Node&, const Node&, Any&, const double, const double);

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

#endif // operations_h
