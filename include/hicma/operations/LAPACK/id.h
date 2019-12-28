#ifndef hicma_operations_LAPACK_id_h
#define hicma_operations_LAPACK_id_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;

std::vector<int> id(Node& A, Node& B, int k);

MULTI_METHOD(
  id_omm, std::vector<int>,
  virtual_<Node>&, virtual_<Node>&,
  int k
);

} // namespace hicma

#endif // hicma_operations_LAPACK_id_h
