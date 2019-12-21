#ifndef hicma_operations_misc_norm_h
#define hicma_operations_misc_norm_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;


namespace hicma
{

class Node;

double norm(const Node&);

MULTI_METHOD(
  norm_omm, double,
  const virtual_<Node>&
);

} // namespace hicma

#endif // hicma_operations_misc_norm_h
