#ifndef operations_h
#define operations_h

#include "any.h"
#include "node.h"

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

void getrf(Any&);

void getrf(Node&);

MULTI_METHOD(
  getrf_omm, void,
  virtual_<Node>&
);

} // namespace hicma

#endif // operations_h
