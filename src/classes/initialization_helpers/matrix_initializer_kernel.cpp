#include "hicma/classes/initialization_helpers/matrix_initializer_kernel.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/initialization_helpers/matrix_initializer.h"
#include "hicma/util/pre_scheduler.h"

#include <cstdint>


namespace hicma
{

MatrixInitializerKernel::MatrixInitializerKernel(
  void (*kernel)(
    Dense& A,
    const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  const std::vector<std::vector<double>>& x,
  int64_t admis, int64_t rank,
  int basis_type
) : MatrixInitializer(admis, rank, basis_type), kernel(kernel), x(x) {}

void MatrixInitializerKernel::fill_dense_representation(
  Dense& A,
  const ClusterTree& node
) const {
  fill_dense_representation(A, node.rows, node.cols);
}

void MatrixInitializerKernel::fill_dense_representation(
  Dense& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  kernel(A, x, row_range.start, col_range.start);
}

Dense MatrixInitializerKernel::get_dense_representation(
  const ClusterTree& node
) const {
  Dense representation(node.rows.n, node.cols.n);
  fill_dense_representation(representation, node.rows, node.cols);
  return representation;
}

} // namespace hicma
