#ifndef hicma_operations_misc_transpose_h
#define hicma_operations_misc_transpose_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

void transpose(Node&);

declare_method(void, transpose_omm, (virtual_<Node&>))

} // namespace hicma

#endif // hicma_operations_misc_transpose_h
