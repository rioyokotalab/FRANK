#ifndef hicma_util_get_memory_usage_h
#define hicma_util_get_memory_usage_h


namespace hicma
{

class Node;

unsigned long get_memory_usage(const Node&, bool include_structure=true);

} // namespace hicma

#endif // hicma_util_get_memory_usage_h
