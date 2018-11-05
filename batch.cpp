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

  void rsvd_batch() {}
}
