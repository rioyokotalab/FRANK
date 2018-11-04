#include "any.h"
#include "low_rank.h"
#include "batch.h"

namespace hicma {

  std::vector<Dense> vecA;
  std::vector<Any*> vecLR;

  void low_rank_push(Any& A, Dense& Aij, int rank) {
    A = LowRank(Aij, rank);
  }

  void low_rank_batch() {}
}
