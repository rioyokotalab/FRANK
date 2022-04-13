#ifndef hicma_extension_headers_tuple_types_h
#define hicma_extension_headers_tuple_types_h


//added instead of the definition below
#include "hicma/classes/dense.h"

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class MatrixProxy;
//class Dense;

// NOTE These typedefs are necessary since yomm macros use commas to parse the
// function signature, so type tuples cannot be defined.
typedef std::tuple<MatrixProxy, MatrixProxy> MatrixPair;
typedef std::tuple<Dense<double>, Dense<double>> DensePair;
typedef std::tuple<Dense<double>, Dense<double>, Dense<double>> DenseTriplet;
typedef std::tuple<Dense<double>, std::vector<int64_t>> DenseIndexSetPair;
typedef std::tuple<MatrixProxy, std::vector<int64_t>> MatrixIndexSetPair;
typedef std::tuple<MatrixProxy, MatrixProxy, MatrixProxy> MatrixTriplet;

typedef std::tuple<double, double> DoublePair;

} // namespace hicma

#endif // hicma_extension_headers_tuple_types_h
