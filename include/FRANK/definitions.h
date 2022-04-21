#ifndef FRANK_definitions_h
#define FRANK_definitions_h

#include <cstdint>
#include <tuple>
#include <vector>


namespace FRANK {

class MatrixProxy;
class Dense;

// NOTE These typedefs are necessary since yomm macros use commas to parse the
// function signature, so type tuples cannot be defined.
typedef std::tuple<MatrixProxy, MatrixProxy> MatrixPair;
typedef std::tuple<Dense, Dense> DensePair;
typedef std::tuple<Dense, Dense, Dense> DenseTriplet;
typedef std::tuple<Dense, std::vector<int64_t>> DenseIndexSetPair;
typedef std::tuple<double, double> DoublePair;

enum class MatrixLayout { RowMajor, ColumnMajor };
enum class Side { Left, Right };
enum class Mode { Upper, Lower };
enum class AdmisType { PositionBased, GeometryBased };

} // namespace FRANK

#endif // FRANK_definitions_h
