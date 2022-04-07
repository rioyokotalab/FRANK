#include "hicma/classes/initialization_helpers/matrix_initializer_function.h"

#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/dense.h"

#include <vector>
#include<cmath>
#include <type_traits>
#include <iostream>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class MatrixInitializerFunction<float>;
template class MatrixInitializerFunction<double>;

template<typename T>
MatrixInitializerFunction<T>::MatrixInitializerFunction(
  void (*func)(
    T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const vec2d<double>& params,
    int64_t row_start, int64_t col_start
  ),
  double admis, int64_t rank, int admis_type, vec2d<double> params
) : MatrixInitializer(admis, rank, admis_type, params),
    func(func) {}

template<typename T>
void MatrixInitializerFunction<T>::fill_dense_representation(
  Matrix& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  // TODO improve error handling
  try {
    Dense<T> matrix = dynamic_cast<Dense<T>&>(A);
    func(&(dynamic_cast<Dense<T>&>(A)), matrix.dim[0], matrix.dim[1], matrix.stride,
	  params, row_range.start, col_range.start);
  }
  catch (std::bad_cast& e) {
    std::cerr<<"Trying to fill either a non-dense matrix or a matrix of the wrong datatype (MatrixInitializerFunction)"<<std::endl;
    std::abort();
  }
}

} // namespace hicma
