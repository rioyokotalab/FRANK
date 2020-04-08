#ifndef hicma_operations_arithmetic_h
#define hicma_operations_arithmetic_h

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

// Addition
Node& operator+=(Node&, const Node&);

} // namespace hicma

#endif // hicma_operations_arithmetic_h
