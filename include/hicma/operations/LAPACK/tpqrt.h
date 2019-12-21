#ifndef hicma_operations_LAPACK_tpqrt_h
#define hicma_operations_LAPACK_tpqrt_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;

void tpqrt(Node&, Node&, Node&);

MULTI_METHOD(
  tpqrt_omm, void,
  virtual_<Node>&,
  virtual_<Node>&,
  virtual_<Node>&
);

} // namespace hicma

#endif // hicma_operations_LAPACK_tpqrt_h
