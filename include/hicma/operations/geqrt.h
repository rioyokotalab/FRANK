#ifndef operations_geqrt_h
#define operations_geqrt_h

#include "hicma/node_proxy.h"
#include "hicma/node.h"

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

void geqrt(NodeProxy&, NodeProxy&);
void geqrt(NodeProxy&, Node&);
void geqrt(Node&, NodeProxy&);

void geqrt(Node&, Node&);

void geqrt2(Dense&, Dense&);

MULTI_METHOD(
  geqrt_omm, void,
  virtual_<Node>&, virtual_<Node>&
);

} // namespace hicma

#endif // operations_geqrt_h
