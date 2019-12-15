#ifndef util_l2_error_h
#define util_l2_error_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;


namespace hicma
{

class Node;

double l2_error(const Node&, const Node&);

MULTI_METHOD(
  l2_error_omm, double,
  const virtual_<Node>&, const virtual_<Node>&
);

} // namespace hicma

#endif // util_l2_error_h
