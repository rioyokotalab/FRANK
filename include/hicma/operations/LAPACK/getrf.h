#ifndef hicma_operations_LAPACK_getrf_h
#define hicma_operations_LAPACK_getrf_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;

void getrf(Node&);

MULTI_METHOD(
  getrf_omm, void,
  virtual_<Node>&
);

} // namespace hicma

#endif // hicma_operations_LAPACK_getrf_h