#ifndef operations_id_h
#define operations_id_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;

std::vector<int> id(Node& A, Node& B, const int k);

MULTI_METHOD(
  id_omm, std::vector<int>,
  virtual_<Node>&, virtual_<Node>&,
  const int k
);

} // namespace hicma

#endif // operations_id_h
