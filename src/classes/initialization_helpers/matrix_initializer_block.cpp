#include "hicma/classes/initialization_helpers/matrix_initializer_block.h"

#include "hicma/extension_headers/classes.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <utility>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class MatrixInitializerBlock<float>;
template class MatrixInitializerBlock<double>;

// Additional constructors
// always position based admissibility
template<typename U>
MatrixInitializerBlock<U>::MatrixInitializerBlock(
  Dense<U>&& matrix, double admis, int64_t rank, int admis_type, vec2d<double> params
) : MatrixInitializer(admis, rank, admis_type, params), matrix(std::move(matrix)) {}


template<typename U>
void MatrixInitializerBlock<U>::fill_dense_representation(
  Matrix& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  fill_dense_from(matrix, A, row_range.start, col_range.start);
}

} // namespace hicma
