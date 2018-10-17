#include "any.h"
#include "low_rank.h"
#include "batch_rsvd.h"

namespace hicma {

  std::vector<int> h_m;
  std::vector<int> h_n;
  std::vector<Dense> vecA;
  std::vector<Any*> vecLR;

  void low_rank_push(Any& A, Dense& Aij, int rank) {
    A = LowRank(Aij, rank);
  }

  void batch_rsvd() {}
}
