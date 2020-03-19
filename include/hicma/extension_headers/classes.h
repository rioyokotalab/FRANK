#ifndef hicma_extension_headers_classes_h
#define hicma_extension_headers_classes_h

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/no_copy_split.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma
{

declare_method(
  Hierarchical, make_hierarchical,
  (virtual_<const Node &>, int, int)
)

declare_method(NoCopySplit, make_no_copy_split, (virtual_<Node&>, int, int))

declare_method(
  NoCopySplit, make_no_copy_split_const,
  (virtual_<const Node&>, int, int)
)

declare_method(Dense, make_dense, (virtual_<const Node&>))

} // namespace hicma

#endif // hicma_extension_headers_classes_h
