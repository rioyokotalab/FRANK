#ifndef hicma_extension_headers_tuple_types_h
#define hicma_extension_headers_tuple_types_h

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class MatrixProxy;
class Dense;

// NOTE These typedefs are necessary since yomm macros use commas to parse the
// function signature, so type tuples cannot be defined.
typedef std::tuple<MatrixProxy, MatrixProxy> MatrixPair;
typedef std::tuple<Dense, Dense> DensePair;
typedef std::tuple<Dense, Dense, Dense> DenseTriplet;
typedef std::tuple<Dense, std::vector<int64_t>> DenseIndexSetPair;

typedef std::tuple<double, double> DoublePair;

} // namespace hicma

#endif // hicma_extension_headers_tuple_types_h
