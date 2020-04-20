#include "hicma/classes/intitialization_helpers/matrix_initializer.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"

#include <cstdint>


namespace hicma
{

MatrixInitializer::MatrixInitializer(
  void (*kernel)(
    Dense& A, const std::vector<double>& x, int64_t row_start, int64_t col_start
  ),
  const std::vector<double>& x
) : kernel(kernel), x(x) {}

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

} // namespace hicma
