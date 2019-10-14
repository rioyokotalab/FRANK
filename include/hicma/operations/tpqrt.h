#ifndef operations_tpqrt_h
#define operations_tpqrt_h

#include "hicma/node_proxy.h"
#include "hicma/node.h"

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

void tpqrt(NodeProxy&, NodeProxy&, NodeProxy&);
void tpqrt(NodeProxy&, NodeProxy&, Node&);
void tpqrt(NodeProxy&, Node&, NodeProxy&);
void tpqrt(NodeProxy&, Node&, Node&);
void tpqrt(Node&, NodeProxy&, NodeProxy&);
void tpqrt(Node&, NodeProxy&, Node&);
void tpqrt(Node&, Node&, NodeProxy&);

void tpqrt(Node&, Node&, Node&);

MULTI_METHOD(
  tpqrt_omm, void,
  virtual_<Node>&,
  virtual_<Node>&,
  virtual_<Node>&
);

} // namespace hicma

#endif // operations_tpqrt_h
