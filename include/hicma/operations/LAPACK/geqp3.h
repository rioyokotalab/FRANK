#ifndef hicma_operations_LAPACK_geqp3_h
#define hicma_operations_LAPACK_geqp3_h

#include "hicma/classes/node.h"

#include <vector>

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma
{

std::vector<int> geqp3(Node& A, Node& R);

declare_method(
  std::vector<int>, geqp3_omm,
  (virtual_<Node&>, virtual_<Node&>)
);

} // namespace hicma

#endif // hicma_operations_LAPACK_geqp3_h
