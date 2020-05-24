#include "hicma/classes/intitialization_helpers/matrix_initializer.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"

#include <cstdint>


namespace hicma
{

MatrixInitializer::MatrixInitializer(
  void (*kernel)(
    Dense& A,
    const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  const std::vector<std::vector<double>>& x,
  int64_t admis, int64_t rank
) : kernel(kernel), x(x), admis(admis), rank(rank) {}

void MatrixInitializer::fill_dense_representation(
  Dense& A,
  const ClusterTree& node
) const {
  kernel(A, x, node.start[0], node.start[1]);
}

Dense MatrixInitializer::get_dense_representation(
  const ClusterTree& node
) const {
  Dense representation(node.dim[0], node.dim[1]);
  kernel(representation, x, node.start[0], node.start[1]);
  return representation;
}

LowRank MatrixInitializer::get_compressed_representation(
  const ClusterTree& node
) const {
  return LowRank(get_dense_representation(node), rank);
}

bool MatrixInitializer::is_admissible(const ClusterTree& node) const {
  bool admissible = true;
  // Main admissibility condition
  admissible &= (node.dist_to_diag() > admis);
  // Vectors are never admissible
  admissible &= (node.dim[0] > 1 && node.dim[1] > 1);
  return admissible;
}

} // namespace hicma
