#ifndef hicma_operations_arithmetic_h
#define hicma_operations_arithmetic_h

#include "hicma/classes/dense.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

// Addition
Node& operator+=(Node&, const Node&);

Dense operator+(const Dense&, const Dense&);

// Subtraction
NodeProxy operator-(const Node&, const Node&);

// Multiplication
Node& operator*=(Node&, double);

} // namespace hicma

#endif // hicma_operations_arithmetic_h
