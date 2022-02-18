#include "hicma/classes/initialization_helpers/matrix_initializer_block.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/extension_headers/util.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <utility>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class MatrixInitializerBlock<float>;
template class MatrixInitializerBlock<double>;

// Additional constructors
template<typename U>
MatrixInitializerBlock<U>::MatrixInitializerBlock(
  Dense<U>&& A, double admis, int64_t rank
) : MatrixInitializer(admis, rank), matrix(std::move(A)) {}


// Utility methods
declare_method(void, fill_dense_from_block, (virtual_<const Matrix&>, virtual_<Matrix&>, int64_t, int64_t))

template<typename U>
void MatrixInitializerBlock<U>::fill_dense_representation(
  Matrix& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  fill_dense_from_block(matrix, A, row_range.start, col_range.start);
}

template<typename U, typename T>
void fill_dense_from_dense(const Dense<U>& matrix, Dense<T>& A, int64_t row_start, int64_t col_start) {
  matrix.copy_to(A, row_start, col_start);
}

define_method(void, fill_dense_from_block, (const Dense<float>& M, Dense<float>& A, int64_t row_start, int64_t col_start)) {
  fill_dense_from_dense(M, A, row_start, col_start);
}

define_method(void, fill_dense_from_block, (const Dense<float>& M, Dense<double>& A, int64_t row_start, int64_t col_start)) {
  fill_dense_from_dense(M, A, row_start, col_start);
}

define_method(void, fill_dense_from_block, (const Dense<double>& M, Dense<float>& A, int64_t row_start, int64_t col_start)) {
  fill_dense_from_dense(M, A, row_start, col_start);
}

define_method(void, fill_dense_from_block, (const Dense<double>& M, Dense<double>& A, int64_t row_start, int64_t col_start)) {
  fill_dense_from_dense(M, A, row_start, col_start);
}

define_method(void, fill_dense_from_block, (const Matrix& M, Matrix& A, int64_t, int64_t)) {
  omm_error_handler("fill_dense_from_block", {M, A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
