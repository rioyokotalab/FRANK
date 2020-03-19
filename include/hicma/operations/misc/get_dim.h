#ifndef hicma_operations_misc_get_dim_h
#define hicma_operations_misc_get_dim_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

int get_n_rows(const Node&);

declare_method(int, get_n_rows_omm, (virtual_<const Node&>))

int get_n_cols(const Node&);

declare_method(int, get_n_cols_omm, (virtual_<const Node&>))

} // namespace hicma

#endif // hicma_operations_misc_get_dim_h
