#ifndef hicma_operations_LAPACK_getrf_h
#define hicma_operations_LAPACK_getrf_h

#include <tuple>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;
class NodeProxy;

typedef std::tuple<NodeProxy, NodeProxy> NodePair;

std::tuple<NodeProxy, NodeProxy> getrf(Node&);

MULTI_METHOD(
  getrf_omm, NodePair,
  virtual_<Node>&
);

} // namespace hicma

#endif // hicma_operations_LAPACK_getrf_h