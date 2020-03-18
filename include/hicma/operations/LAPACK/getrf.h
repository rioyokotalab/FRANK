#ifndef hicma_operations_LAPACK_getrf_h
#define hicma_operations_LAPACK_getrf_h

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"

#include <tuple>

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma
{

typedef std::tuple<NodeProxy, NodeProxy> NodePair;

std::tuple<NodeProxy, NodeProxy> getrf(Node&);

declare_method(
  NodePair, getrf_omm,
  (virtual_<Node&>)
);

} // namespace hicma

#endif // hicma_operations_LAPACK_getrf_h