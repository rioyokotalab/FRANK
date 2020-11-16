#ifndef hicma_util_geometry_file_h
#define hicma_util_geometry_file_h

#include <string>
#include <vector>


namespace hicma
{

std::vector<std::vector<double>> read_geometry_file(std::string filename);

} // namespace hicma

#endif // hicma_util_experiment_geometry_file_h
