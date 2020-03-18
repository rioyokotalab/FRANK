#ifndef hicma_operations_LAPACK_trsm_h
#define hicma_operations_LAPACK_trsm_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma
{

// TODO consider splitting left and right trsm into separate functions! That
// would allow nicer syntax: ltrsm(Tri, A, 'u/l') and rtrsm(A, Tri, 'u/l').
void trsm(const Node&, Node&, const char& uplo, bool left=true);

declare_method(
  void, trsm_omm,
  (virtual_<const Node&>, virtual_<Node&>, const char&, bool)
);

} // namespace hicma

#endif // hicma_operations_LAPACK_trsm_h