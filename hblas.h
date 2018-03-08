#ifndef hblas_h
#define hblas_h
#include "node.h"
#include "dense.h"
#include "low_rank.h"
#include "hierarchical.h"

#define DEBUG 1

namespace hicma {
  Dense& D_t(const boost::any& A);

  LowRank& L_t(const boost::any& A);

  Hierarchical& H_t(const boost::any& A);

  void getrf(boost::any& A);

  void trsm(const boost::any& Aii, boost::any& Aij, const char& uplo);

  void gemm(const boost::any& A, const boost::any& B, boost::any& C);

  void add(const boost::any& A, const boost::any& B, boost::any& C);

  void sub(const boost::any& A, const boost::any& B, boost::any& C);

  double norm(boost::any& A);

  void assign(const boost::any& A, const double a);
}
#endif
