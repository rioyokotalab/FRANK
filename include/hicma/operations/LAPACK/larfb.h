#ifndef hicma_operations_LAPACK_larfb_h
#define hicma_operations_LAPACK_larfb_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;

void larfb(const Node&, const Node&, Node&, const bool);

MULTI_METHOD(
  larfb_omm, void,
  const virtual_<Node>&,
  const virtual_<Node>&,
  virtual_<Node>&,
  const bool
);

} // namespace hicma

#endif // hicma_operations_LAPACK_larfb_h