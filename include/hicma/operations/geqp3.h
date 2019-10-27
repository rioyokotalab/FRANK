#ifndef operations_geqp3_h
#define operations_geqp3_h

#include <vector>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;

std::vector<int> geqp3(Node&, Node&);

MULTI_METHOD(
  geqp3_omm, std::vector<int>,
  virtual_<Node>&,
  virtual_<Node>&
);

} // namespace hicma

#endif // operations_geqp3_h
