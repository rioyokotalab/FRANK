#ifndef FRANK_util_geometry_file_h
#define FRANK_util_geometry_file_h

#include <string>
#include <vector>


namespace FRANK
{

/**
 * @brief Reads the geometry information from a text file
 * 
 * @param filename location of the file to be read
 * @return std::vector<std::vector<double>> matrix of vertices representing the geometry
 *
 * Geometry file format
 * ```
 * n dim
 * x1 y1 z1
 * x2 y2 z2
 * ...
 * xn yn zn
 * ```
 */
std::vector<std::vector<double>> read_geometry_file(const std::string filename);

} // namespace FRANK

#endif // FRANK_util_experiment_geometry_file_h
