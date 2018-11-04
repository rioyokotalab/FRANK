#ifndef batch_rsvd_h
#define batch_rsvd_h

#include <vector>

namespace hicma {

  class Any;
  class Dense;

  extern std::vector<Dense> vecA;
  extern std::vector<Any*> vecLR;

  void low_rank_push(Any& A, Dense& Aij, int rank);

  void low_rank_batch();
}

#endif
