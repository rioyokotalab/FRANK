#ifndef operations_getrf_h
#define operations_getrf_h

#include "hicma/node_proxy.h"
#include "hicma/node.h"

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

void getrf(NodeProxy&);

void getrf(Node&);

MULTI_METHOD(
  getrf_omm, void,
  virtual_<Node>&
);

} // namespace hicma

#endif