#include "hicma/classes/initialization_helpers/matrix_initializer_block.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <utility>


namespace hicma
{

// Additional constructors
MatrixInitializerBlock::MatrixInitializerBlock(
  Dense&& A, int64_t admis, int64_t rank
) : MatrixInitializer(admis, rank), matrix(std::move(A)) {}

// Utility methods
void MatrixInitializerBlock::fill_dense_representation(
  Dense& A, const ClusterTree& node
) const {
  fill_dense_representation(A, node.rows, node.cols);
}

void MatrixInitializerBlock::fill_dense_representation(
  Dense& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      A(i, j) = matrix(row_range.start+i, col_range.start+j);
    }
  }
}

Dense MatrixInitializerBlock::get_dense_representation(
  const ClusterTree& node
) const {
  Dense out(node.rows.n, node.cols.n);
  fill_dense_representation(out, node.rows, node.cols);
  return out;
}

} // namespace hicma
