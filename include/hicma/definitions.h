#ifndef hicma_definitions_h
#define hicma_definitions_h

#include <vector>


namespace hicma {

enum class MatrixLayout { RowMajor, ColumnMajor };
enum class Side { Left, Right };
enum class Mode { Upper, Lower };
enum class AdmisType { PositionBased, GeometryBased };

} // namespace hicma

#endif // hicma_definitions_h
