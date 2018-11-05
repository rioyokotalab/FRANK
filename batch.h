#ifndef batch_h
#define batch_h

#include <vector>

namespace hicma {

  class Any;
  class Dense;

  extern std::vector<Dense> vecA;
  extern std::vector<Dense> vecB;
  extern std::vector<Dense*> vecC;
  extern std::vector<Any*> vecLR;

  void rsvd_push(Any& A, Dense& Aij, int rank);

  void gemm_push(Dense& A, Dense& B, Dense* C);

  void rsvd_batch();

  void gemm_batch();
}

#endif
