#ifndef hicma_operations_misc_norm_h
#define hicma_operations_misc_norm_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

double norm(const Node&);

declare_method(double, norm_omm, (virtual_<const Node&>))

} // namespace hicma

#endif // hicma_operations_misc_norm_h
