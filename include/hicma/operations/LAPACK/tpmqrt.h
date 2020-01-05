#ifndef hicma_operations_LAPACK_tpmqrt_h
#define hicma_operations_LAPACK_tpmqrt_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;

void tpmqrt(
  const Node&, const Node&, Node&, Node&,
  bool
);

MULTI_METHOD(
  tpmqrt_omm, void,
  const virtual_<Node>&, const virtual_<Node>&,
  virtual_<Node>&, virtual_<Node>&,
  bool
);

} // namespace hicma

#endif // hicma_operations_LAPACK_tpmqrt_h
