#ifndef hicma_extension_headers_operations_h
#define hicma_extension_headers_operations_h

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/dense.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <vector>


namespace hicma
{

declare_method(
  void, gemm_omm,
  (
    virtual_<const Node&>, virtual_<const Node&>, virtual_<Node&>,
    double, double
  )
);

declare_method(
  void, trmm_omm,
  (
    virtual_<const Node&>, virtual_<Node&>,
    const char&, const char&, const char&, const char&,
    const double&
  )
);

declare_method(
  void, trsm_omm,
  (virtual_<const Node&>, virtual_<Node&>, const char&, bool)
);

} // namespace hicma

#endif // hicma_extension_headers_operations_h
