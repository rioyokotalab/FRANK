#ifndef FRANK_util_geometry_file_h
#define FRANK_util_geometry_file_h

#include <string>
#include <vector>


namespace FRANK
{

/**
 * @brief reads the geometry information from a file
 * 
 * The first two values read must contain the number of rows and colums to be read.
 * I.e:
 * n_rows n_cols value value value ...
 * 
 * @param filename location of the file to be read
 * @return std::vector<std::vector<double>> matrix of vertices representing the geometry
 */
std::vector<std::vector<double>> read_geometry_file(const std::string filename);

} // namespace FRANK

#endif // FRANK_util_experiment_geometry_file_h
