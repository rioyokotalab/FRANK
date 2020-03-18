#ifndef hicma_operations_misc_addition_h
#define hicma_operations_misc_addition_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

void operator+=(Node&, const Node&);

declare_method(
  void, addition_omm,
  (virtual_<Node&>, virtual_<const Node&>)
);

} // namespace hicma

#endif // hicma_operations_misc_addition_h
