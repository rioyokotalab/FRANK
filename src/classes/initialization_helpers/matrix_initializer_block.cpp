#include "hicma/classes/initialization_helpers/matrix_initializer_block.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/extension_headers/util.h"

#include <utility>
#include <typeinfo>
#include <iostream>


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
template<typename U>
void MatrixInitializerBlock<U>::fill_dense_representation(
  Matrix& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  try {
    if (is_double(A)){
      matrix.copy_to(dynamic_cast<Dense<double>&>(A), row_range.start, col_range.start);
    }
    else {
      matrix.copy_to(dynamic_cast<Dense<float>&>(A), row_range.start, col_range.start);
    }
  }
  catch(std::bad_cast& e) {
    // TODO better error handling
    std::cerr<<"MatrixInitializerBlock: Could not initialize a non dense matrix."<<std::endl;
    std::abort();
  }
}

} // namespace hicma
