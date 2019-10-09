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


void trsm(const Any&, Any&, const char& uplo);
void trsm(const Any&, Node&, const char& uplo);
void trsm(const Node&, Any&, const char& uplo);

void trsm(const Node&, Node&, const char& uplo);

MULTI_METHOD(
  trsm_omm, void,
  const virtual_<Node>&,
  virtual_<Node>&,
  const char& uplo
);
} // namespace hicma

#endif // operations_h
