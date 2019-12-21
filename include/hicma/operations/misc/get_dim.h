#ifndef hicma_operations_misc_get_dim_h
#define hicma_operations_misc_get_dim_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;


namespace hicma
{

class Node;

int get_n_rows(const Node&);

MULTI_METHOD(
  get_n_rows_omm, int,
  const virtual_<Node>&
);

int get_n_cols(const Node&);

MULTI_METHOD(
  get_n_cols_omm, int,
  const virtual_<Node>&
);

} // namespace hicma

#endif // hicma_operations_misc_get_dim_h
