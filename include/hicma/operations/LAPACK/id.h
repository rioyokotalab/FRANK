#ifndef hicma_operations_LAPACK_id_h
#define hicma_operations_LAPACK_id_h

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"

#include <vector>

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma
{

std::vector<int> id(Node& A, Node& B, int k);

declare_method(
  std::vector<int>, id_omm,
  (virtual_<Node&>, virtual_<Node&>, int)
);

Dense get_cols(const Dense& A, std::vector<int> P);

std::tuple<Dense, Dense, Dense> two_sided_id(Node& A, int k);

typedef std::tuple<Dense, Dense, Dense> dense_triplet;
declare_method(dense_triplet, two_sided_id_omm, (virtual_<Node&>, int));

} // namespace hicma

#endif // hicma_operations_LAPACK_id_h
