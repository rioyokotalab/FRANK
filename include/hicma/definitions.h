#ifndef hicma_definitions_h
#define hicma_definitions_h

#include <vector>

template <typename U>
using vec2d = std::vector<std::vector<U>>;

enum MatrixLayout { HICMA_ROW_MAJOR, HICMA_COL_MAJOR };
//To be removed if admissibility checker is passed as function
enum { POSITION_BASED_ADMIS, GEOMETRY_BASED_ADMIS };

#endif // hicma_definitions_h
