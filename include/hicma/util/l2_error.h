#ifndef hicma_util_l2_error_h
#define hicma_util_l2_error_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

double l2_error(const Node&, const Node&);

declare_method(
  double, l2_error_omm,
  (virtual_<const Node&>, virtual_<const Node&>)
)

} // namespace hicma

#endif // hicma_util_l2_error_h
