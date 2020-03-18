#ifndef hicma_operations_LAPACK_tpqrt_h
#define hicma_operations_LAPACK_tpqrt_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma
{

void tpqrt(Node&, Node&, Node&);

declare_method(
  void, tpqrt_omm,
  (virtual_<Node&>, virtual_<Node&>, virtual_<Node&>)
);

} // namespace hicma

#endif // hicma_operations_LAPACK_tpqrt_h
