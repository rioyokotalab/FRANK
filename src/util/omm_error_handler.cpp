#include "hicma/util/omm_error_handler.h"

#include "hicma/classes/node.h"
#include "hicma/util/print.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>


namespace hicma
{

void omm_error_handler(
  const char* omm_name,
  std::vector<std::reference_wrapper<const Matrix>> virtual_arguments,
  const char* file, int line
) {
  std::cerr << omm_name << "(";
  if (virtual_arguments.size() > 0) {
    std::cerr << type(virtual_arguments[0].get());
    for (size_t i=1; i<virtual_arguments.size(); ++i) {
      std::cerr << ", " << type(virtual_arguments[i].get());
    }
  }
  std::cerr << ") undefined! (" << file << ":" << line << ")" << std::endl;
  std::cerr << "Note that type information here relies on providing a ";
  std::cerr << "specialization of type_omm() for any new classes!" << std::endl;
}

} // namespace hicma
