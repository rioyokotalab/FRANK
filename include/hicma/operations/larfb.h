#ifndef operations_larfb_h
#define operations_larfb_h

#include "hicma/node_proxy.h"
#include "hicma/node.h"

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

void larfb(const NodeProxy&, const NodeProxy&, NodeProxy&, const bool);
void larfb(const NodeProxy&, const NodeProxy&, Node&, const bool);
void larfb(const NodeProxy&, const Node&, NodeProxy&, const bool);
void larfb(const NodeProxy&, const Node&, Node&, const bool);
void larfb(const Node&, const NodeProxy&, NodeProxy&, const bool);
void larfb(const Node&, const NodeProxy&, Node&, const bool);
void larfb(const Node&, const Node&, NodeProxy&, const bool);

void larfb(const Node&, const Node&, Node&, const bool);

MULTI_METHOD(
  larfb_omm, void,
  const virtual_<Node>&,
  const virtual_<Node>&,
  virtual_<Node>&,
  const bool
);

} // namespace hicma


#endif // operations_larfb_h