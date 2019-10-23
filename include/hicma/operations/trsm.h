#ifndef operations_trsm_h
#define operations_trsm_h

#include "hicma/node_proxy.h"
#include "hicma/node.h"

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

void trsm(const NodeProxy&, NodeProxy&, const char& uplo);
void trsm(const NodeProxy&, Node&, const char& uplo);
void trsm(const Node&, NodeProxy&, const char& uplo);

void trsm(const Node&, Node&, const char& uplo);

MULTI_METHOD(
  trsm_omm, void,
  const virtual_<Node>&,
  virtual_<Node>&,
  const char& uplo
);

} // namespace hicma

#endif