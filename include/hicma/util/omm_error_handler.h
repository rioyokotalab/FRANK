#ifndef hicma_util_omm_error_handler_h
#define hicma_util_omm_error_handler_h

#include <functional>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

/**
 * @brief Error-handler for Open Multi-Methods
 * 
 * Invoked if a `YOMM2` function cannot find a requiered specialization.
 * Displays the name of the function and the arguments provided as well
 * as the corresponding file and line number.
 * 
 * @param omm_name name of the `YOMM2` function
 * @param virtual_arguments parameters for the `YOMM2` function (i.e. Matrix types)
 * @param file name of the file where the function is specified
 * @param line line number where the function is specified.
 */
void omm_error_handler(
  const char* omm_name,
  const std::vector<std::reference_wrapper<const Matrix>> virtual_arguments,
  const char* file, const int line
);

} // namespace hicma

#endif // hicma_util_omm_error_handler_h
