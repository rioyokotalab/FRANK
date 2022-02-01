#include "hicma/classes/initialization_helpers/matrix_initializer_block.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <utility>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class MatrixInitializerBlock<double>;
template class MatrixInitializerBlock<float>;

// Additional constructors
template<typename T>
MatrixInitializerBlock<T>::MatrixInitializerBlock(
  Dense<T>&& A, double admis, int64_t rank
) : MatrixInitializer<T>(admis, rank), matrix(std::move(A)) {}

// Utility methods
template<typename T>
void MatrixInitializerBlock<T>::fill_dense_representation(
  Dense<T>& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      A(i, j) = matrix(row_range.start+i, col_range.start+j);
    }
  }
}

} // namespace hicma
