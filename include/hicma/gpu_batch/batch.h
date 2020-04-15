#ifndef hicma_batch_h
#define hicma_batch_h

#include <cstdint>
#include <vector>


namespace hicma
{

  class NodeProxy;
  class Dense;

  extern std::vector<Dense> vecA;
  extern std::vector<Dense> vecB;
  extern std::vector<Dense*> vecC;
  extern std::vector<NodeProxy*> vecLR;

  void rsvd_push(NodeProxy& A, Dense& Aij, int64_t rank);

  void gemm_push(const Dense& A, const Dense& B, Dense& C);

  void rsvd_batch();

  void gemm_batch();

} // namespace hicma

#endif // hicma_batch_h
