#ifndef hicma_operations_LAPACK_trsm_h
#define hicma_operations_LAPACK_trsm_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;

void trsm(const Node&, Node&, const char& uplo);

MULTI_METHOD(
  trsm_omm, void,
  const virtual_<Node>&,
  virtual_<Node>&,
  const char& uplo
);

} // namespace hicma

#endif // hicma_operations_LAPACK_trsm_h