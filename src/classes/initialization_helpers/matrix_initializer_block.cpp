#include "FRANK/classes/initialization_helpers/matrix_initializer_block.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/initialization_helpers/cluster_tree.h"
#include "FRANK/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <utility>


namespace FRANK
{

// Additional constructors
MatrixInitializerBlock::MatrixInitializerBlock(
  Dense&& A, const double admis, const double eps, const int64_t rank,
  const std::vector<std::vector<double>> params, const AdmisType admis_type
) : MatrixInitializer(admis, eps, rank, params, admis_type),
    matrix(std::move(A)) {}

// Utility methods
void MatrixInitializerBlock::fill_dense_representation(
  Dense& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      A(i, j) = matrix(row_range.start+i, col_range.start+j);
    }
  }
}

} // namespace FRANK
