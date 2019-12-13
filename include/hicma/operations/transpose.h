#ifndef operations_transpose_h
#define operations_transpose_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;


namespace hicma
{

class Node;

void transpose(Node&);

MULTI_METHOD(
  transpose_omm, void,
  virtual_<Node>&
);

} // namespace hicma

#endif // operations_transpose_h
