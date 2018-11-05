#include "any.h"
#include "low_rank.h"
#include "batch.h"

namespace hicma {

  std::vector<Dense> vecA;
  std::vector<Dense> vecB;
  std::vector<Dense*> vecC;
  std::vector<Any*> vecLR;

  void rsvd_push(Any& A, Dense& Aij, int rank) {
    A = LowRank(Aij, rank);
  }

  void gemm_push(const Dense& A, const Dense& B, Dense* C) {
    C->gemm(A, B, CblasNoTrans, CblasNoTrans, 1, 1);
  }

  void rsvd_batch() {}

  void gemm_batch() {}
}
