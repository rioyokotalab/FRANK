#ifndef hicma_util_l2_error_h
#define hicma_util_l2_error_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

double l2_error(const Node&, const Node&);

typedef std::tuple<double, double> DoublePair;
declare_method(
  DoublePair, collect_diff_norm_omm,
  (virtual_<const Node&>, virtual_<const Node&>)
)

} // namespace hicma

#endif // hicma_util_l2_error_h
