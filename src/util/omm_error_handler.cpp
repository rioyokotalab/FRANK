#include "hicma/util/omm_error_handler.h"

#include "hicma/classes/node.h"

#include <functional>
#include <iostream>
#include <vector>


namespace hicma
{

void omm_error_handler(
  const char* omm_name,
  std::vector<std::reference_wrapper<const Node>> virtual_arguments,
  const char* file, int line
) {
  std::cerr << omm_name << "(";
  if (virtual_arguments.size() > 0) {
    std::cerr << virtual_arguments[0].get().type();
    for (unsigned int i=1; i<virtual_arguments.size(); ++i) {
      std::cerr << ", " << virtual_arguments[i].get().type();
    }
  }
  std::cerr << ") undefined! (" << file << ":" << line << ")" << std::endl;
  std::cerr << "Note that type information here relies on overriding the ";
  std::cerr << "Node.type() function in any participating class!" << std::endl;
}

} // namespace hicma
