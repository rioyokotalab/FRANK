#ifndef batch_h
#define batch_h

#include <vector>

namespace hicma {

  class NodeProxy;
  class Dense;

  extern std::vector<Dense> vecA;
  extern std::vector<Dense> vecB;
  extern std::vector<Dense*> vecC;
  extern std::vector<NodeProxy*> vecLR;

  void rsvd_push(NodeProxy& A, Dense& Aij, int rank);

  void gemm_push(const Dense& A, const Dense& B, Dense& C);

  void rsvd_batch();

  void gemm_batch();
}

#endif
