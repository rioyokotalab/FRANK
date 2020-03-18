#ifndef hicma_util_get_memory_usage_h
#define hicma_util_get_memory_usage_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

unsigned long get_memory_usage(const Node&, bool include_structure=true);

declare_method(
  unsigned long, get_memory_usage_omm,
  (virtual_<const Node&>, bool)
);

} // namespace hicma

#endif // hicma_util_get_memory_usage_h
