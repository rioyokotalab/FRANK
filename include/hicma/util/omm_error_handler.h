#ifndef hicma_util_omm_error_handler_h
#define hicma_util_omm_error_handler_h

#include <functional>
#include <vector>


namespace hicma
{

class Node;

void omm_error_handler(
  const char* omm_name,
  std::vector<std::reference_wrapper<const Node>> virtual_arguments,
  const char* file, int line
);

} // namespace hicma

#endif // hicma_util_omm_error_handler_h
