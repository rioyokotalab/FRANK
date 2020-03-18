#ifndef hicma_operations_LAPACK_geqrt_h
#define hicma_operations_LAPACK_geqrt_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma
{

class Dense;

void geqrt(Node&, Node&);

void geqrt2(Dense&, Dense&);

declare_method(
  void, geqrt_omm,
  (virtual_<Node&>, virtual_<Node&>)
);

} // namespace hicma

#endif // hicma_operations_LAPACK_geqrt_h
