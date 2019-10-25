#ifndef operations_tpmqrt_h
#define operations_tpmqrt_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;

void tpmqrt(
  const Node&, const Node&, Node&, Node&,
  const bool
);

MULTI_METHOD(
  tpmqrt_omm, void,
  const virtual_<Node>&, const virtual_<Node>&,
  virtual_<Node>&, virtual_<Node>&,
  const bool
);

} // namespace hicma

#endif // operations_tpmqrt_h
