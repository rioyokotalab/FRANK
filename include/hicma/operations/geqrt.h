#ifndef operations_geqrt_h
#define operations_geqrt_h

#include "hicma/node_proxy.h"
#include "hicma/node.h"

namespace hicma
{

void geqrt(NodeProxy&, NodeProxy&);
void geqrt(NodeProxy&, Node&);
void geqrt(Node&, NodeProxy&);

void geqrt(Node&, Node&);

void geqrt2(Dense&, Dense&);

} // namespace hicma


#endif // operations_geqrt_h
