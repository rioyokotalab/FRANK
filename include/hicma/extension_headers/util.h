#ifndef hicma_extension_headers_util_h
#define hicma_extension_headers_util_h

#include "hicma/classes/node.h"
#include "hicma/extension_headers/tuple_types.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

declare_method(
  unsigned long, get_memory_usage_omm, (virtual_<const Node&>, bool))

declare_method(
  DoublePair, collect_diff_norm_omm,
  (virtual_<const Node&>, virtual_<const Node&>)
)

declare_method(void, print_omm, (virtual_<const Node&>))

} // namespace hicma

#endif // hicma_extension_headers_util_h
