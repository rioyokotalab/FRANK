#ifndef operations_tpqrt_h
#define operations_tpqrt_h

#include "hicma/node_proxy.h"
#include "hicma/node.h"

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

} // namespace hicma

#endif // operations_tpqrt_h
