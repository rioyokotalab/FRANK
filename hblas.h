#ifndef hblas_h
#define hblas_h
#include "dense.h"
#include "low_rank.h"
#include "hierarchical.h"

namespace hicma {
  std::vector<int> getrf(boost::any& A);

  void trsm(const boost::any& Aii, boost::any& Aij, const char& uplo);

  void gemv(const boost::any& A, const boost::any& b, boost::any& x);

  void gemm(const boost::any& A, const boost::any& B, boost::any& C);

  void add(const boost::any& A, const boost::any& B, boost::any& C);

  void sub(const boost::any& A, const boost::any& B, boost::any& C);
}
#endif
