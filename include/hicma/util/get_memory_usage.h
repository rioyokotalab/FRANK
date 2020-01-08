#ifndef hicma_util_get_memory_usage_h
#define hicma_util_get_memory_usage_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;


namespace hicma
{

class Node;

unsigned long get_memory_usage(const Node&, bool include_structure=true);

MULTI_METHOD(
  get_memory_usage_omm, unsigned long,
  const virtual_<Node>&, bool
);

} // namespace hicma

#endif // hicma_util_get_memory_usage_h
