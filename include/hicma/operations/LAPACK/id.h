#ifndef hicma_operations_LAPACK_id_h
#define hicma_operations_LAPACK_id_h

#include <vector>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class Node;
class Dense;

std::vector<int> id(Node& A, Node& B, int k);

MULTI_METHOD(
  id_omm, std::vector<int>,
  virtual_<Node>&, virtual_<Node>&,
  int k
);

Dense get_cols(const Dense& A, std::vector<int> P);

std::tuple<Dense, Dense, Dense> two_sided_id(Node& A, int k);

typedef std::tuple<Dense, Dense, Dense> dense_triplet;
MULTI_METHOD(
  two_sided_id_omm, dense_triplet,
  virtual_<Node>&, int k
);

} // namespace hicma

#endif // hicma_operations_LAPACK_id_h
