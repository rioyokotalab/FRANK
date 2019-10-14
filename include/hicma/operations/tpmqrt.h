#ifndef operations_tpmqrt_h
#define operations_tpmqrt_h

#include "hicma/node_proxy.h"
#include "hicma/node.h"

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

void tpmqrt(
  const NodeProxy&, const NodeProxy&, NodeProxy&, NodeProxy&,
  const bool
);
void tpmqrt(
  const NodeProxy&, const NodeProxy&, NodeProxy&, Node&,
  const bool
);
void tpmqrt(
  const NodeProxy&, const NodeProxy&, Node&, NodeProxy&,
  const bool
);
void tpmqrt(
  const NodeProxy&, const NodeProxy&, Node&, Node&,
  const bool
);
void tpmqrt(
  const NodeProxy&, const Node&, NodeProxy&, NodeProxy&,
  const bool
);
void tpmqrt(
  const NodeProxy&, const Node&, NodeProxy&, Node&,
  const bool
);
void tpmqrt(
  const NodeProxy&, const Node&, Node&, NodeProxy&,
  const bool
);
void tpmqrt(
  const NodeProxy&, const Node&, Node&, Node&,
  const bool
);
void tpmqrt(
  const Node&, const NodeProxy&, NodeProxy&, NodeProxy&,
  const bool
);
void tpmqrt(
  const Node&, const NodeProxy&, NodeProxy&, Node&,
  const bool
);
void tpmqrt(
  const Node&, const NodeProxy&, Node&, NodeProxy&,
  const bool
);
void tpmqrt(
  const Node&, const NodeProxy&, Node&, Node&,
  const bool
);
void tpmqrt(
  const Node&, const Node&, NodeProxy&, NodeProxy&,
  const bool
);
void tpmqrt(
  const Node&, const Node&, NodeProxy&, Node&,
  const bool
);
void tpmqrt(
  const Node&, const Node&, Node&, NodeProxy&,
  const bool
);

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