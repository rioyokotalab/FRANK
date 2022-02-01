#include "hicma/classes/initialization_helpers/matrix_initializer_kernel.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include <cstdint>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class MatrixInitializerKernel<double>;
template class MatrixInitializerKernel<float>;

template<typename T>
MatrixInitializerKernel<T>::MatrixInitializerKernel(
  void (*kernel)(
    T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const std::vector<std::vector<double>>& params,
    int64_t row_start, int64_t col_start
  ),
  std::vector<std::vector<double>> params,
  double admis, int64_t rank, int admis_type
) : MatrixInitializer<T>(admis, rank, params, admis_type),
    kernel(kernel) {}

template<typename T>
void MatrixInitializerKernel<T>::fill_dense_representation(
  Dense<T>& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  kernel(&A, A.dim[0], A.dim[1], A.stride,
	 this->params, row_range.start, col_range.start);
}

} // namespace hicma
