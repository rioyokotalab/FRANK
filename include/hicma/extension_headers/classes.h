#ifndef hicma_extension_headers_classes_h
#define hicma_extension_headers_classes_h

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/index_range.h"
#include "hicma/classes/node.h"
#include "hicma/classes/no_copy_split.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

declare_method(
  void, fill_hierarchical_from,
  (Hierarchical&, virtual_<const Node &>)
)

declare_method(NoCopySplit, make_no_copy_split, (virtual_<Node&>, int, int))

declare_method(
  NodeProxy, make_view,
  (const IndexRange&, const IndexRange&, virtual_<Node&>)
)
declare_method(
  NodeProxy, make_view,
  (const IndexRange&, const IndexRange&, virtual_<const Node&>)
)

declare_method(void, fill_dense_from, (virtual_<const Node&>, virtual_<Node&>))

} // namespace hicma

#endif // hicma_extension_headers_classes_h
