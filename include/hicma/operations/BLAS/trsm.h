#ifndef hicma_operations_LAPACK_trsm_h
#define hicma_operations_LAPACK_trsm_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;

// TODO consider splitting left and right trsm into separate functions! That
// would allow nicer syntax: ltrsm(Tri, A, 'u/l') and rtrsm(A, Tri, 'u/l').
void trsm(const Node&, Node&, const char& uplo, bool left=true);

MULTI_METHOD(
  trsm_omm, void,
  const virtual_<Node>&,
  virtual_<Node>&,
  const char& uplo,
  bool left
);

} // namespace hicma

#endif // hicma_operations_LAPACK_trsm_h