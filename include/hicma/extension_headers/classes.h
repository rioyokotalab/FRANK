#ifndef hicma_extension_headers_classes_h
#define hicma_extension_headers_classes_h

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/no_copy_split.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>


namespace hicma
{

declare_method(
  void, fill_hierarchical_from,
  (Hierarchical&, virtual_<const Node &>, const ClusterTree&)
)

declare_method(
  NoCopySplit, make_no_copy_split, (virtual_<Node&>, int64_t, int64_t)
)

declare_method(
  NodeProxy, make_view, (const ClusterTree&, virtual_<Node&>)
)
declare_method(
  NodeProxy, make_view, (const ClusterTree&, virtual_<const Node&>)
)

declare_method(void, fill_dense_from, (virtual_<const Node&>, virtual_<Node&>))

} // namespace hicma

#endif // hicma_extension_headers_classes_h
