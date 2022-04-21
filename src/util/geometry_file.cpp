#include "FRANK/util/geometry_file.h"

#include <cstdint>
#include <vector>
#include <fstream>


namespace FRANK
{

std::vector<std::vector<double>> read_geometry_file(const std::string filename) {
  std::ifstream file;
  file.open(filename);
  int64_t n, dim;
  file >>n >>dim;
  std::vector<std::vector<double>> vertices(dim, std::vector<double>());
  double a;
  for(int64_t i=0; i<n; i++) {
    for(int64_t j=0; j<dim; j++) {
      file >>a;
      vertices[j].push_back(a);
    }
  }
  file.close();
  return vertices;
}

} // namespace FRANK
