#ifndef hicma_definitions_h
#define hicma_definitions_h

#include <vector>

typedef std::vector<std::vector<double>> vec2d;


namespace hicma {

enum MatrixLayout { RowMajor, ColumnMajor };
enum Mode { Upper, Lower };
enum Side { Left, Right };
enum AdmisType { PositionBasedAdmis, GeometryBasedAdmis };

} // namespace hicma

#endif // hicma_definitions_h
