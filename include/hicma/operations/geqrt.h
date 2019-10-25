#ifndef operations_geqrt_h
#define operations_geqrt_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;
class Dense;

void geqrt(Node&, Node&);

void geqrt2(Dense&, Dense&);

MULTI_METHOD(
  geqrt_omm, void,
  virtual_<Node>&, virtual_<Node>&
);

} // namespace hicma

#endif // operations_geqrt_h
