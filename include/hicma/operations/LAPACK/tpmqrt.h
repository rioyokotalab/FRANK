#ifndef hicma_operations_LAPACK_tpmqrt_h
#define hicma_operations_LAPACK_tpmqrt_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma
{

void tpmqrt(
  const Node&, const Node&, Node&, Node&,
  bool
);

declare_method(
  void, tpmqrt_omm,
  (
    virtual_<const Node&>, virtual_<const Node&>,
    virtual_<Node&>, virtual_<Node&>,
    bool
  )
);

} // namespace hicma

#endif // hicma_operations_LAPACK_tpmqrt_h
