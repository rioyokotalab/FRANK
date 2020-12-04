#ifndef hicma_definitions_h
#define hicma_definitions_h

#include <vector>

typedef std::vector<std::vector<double>> vec2d;

enum BasisType { NORMAL_BASIS, SHARED_BASIS };
enum MatrixLayout { HICMA_ROW_MAJOR, HICMA_COL_MAJOR };

//To be removed if admissibility checker is passed as function
enum { POSITION_BASED_ADMIS, GEOMETRY_BASED_ADMIS };

#endif // hicma_definitions_h
