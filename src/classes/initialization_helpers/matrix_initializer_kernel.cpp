#include "FRANK/classes/initialization_helpers/matrix_initializer_kernel.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/initialization_helpers/cluster_tree.h"
#include "FRANK/classes/initialization_helpers/matrix_initializer.h"

#include <cstdint>


namespace FRANK
{

MatrixInitializerKernel::MatrixInitializerKernel(
  void (*kernel)(
    double* A, const uint64_t A_rows, const uint64_t A_cols, const uint64_t A_stride,
    const std::vector<std::vector<double>>& params,
    const int64_t row_start, const int64_t col_start
  ),
  const std::vector<std::vector<double>> params,
  const double admis, const double eps, const int64_t rank, const AdmisType admis_type
) : MatrixInitializer(admis, eps, rank, params, admis_type),
    kernel(kernel) {}

void MatrixInitializerKernel::fill_dense_representation(
  Dense& A, const IndexRange& row_range, const IndexRange& col_range
) const {
  kernel(&A, A.dim[0], A.dim[1], A.stride,
         params, row_range.start, col_range.start);
}

} // namespace FRANK
