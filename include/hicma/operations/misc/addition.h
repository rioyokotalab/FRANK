#ifndef hicma_operations_misc_addition_h
#define hicma_operations_misc_addition_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;


namespace hicma
{

class Node;

void operator+=(Node&, const Node&);

MULTI_METHOD(
  addition_omm, void,
  virtual_<Node>&, const virtual_<Node>&
);

} // namespace hicma

#endif // hicma_operations_misc_addition_h
