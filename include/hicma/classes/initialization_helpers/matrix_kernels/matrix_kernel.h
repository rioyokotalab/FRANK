#ifndef hicma_classes_initialization_helpers_matrix_kernels_matrix_kernel_h
#define hicma_classes_initialization_helpers_matrix_kernels_matrix_kernel_h

#include "hicma/definitions.h"

#include <cstdint>
#include <memory>


namespace hicma
{

class IndexRange;
class Matrix;

template<typename U = double>
class MatrixKernel {
  public:
  MatrixKernel() = default;
  MatrixKernel(const MatrixKernel& kernel) = default;
  MatrixKernel(MatrixKernel&& kernel) = default;
  ~MatrixKernel() = default;

  virtual std::unique_ptr<MatrixKernel<U>> clone() const = 0;

  virtual void apply(Matrix& A, int64_t row_start=0, int64_t col_start=0) const = 0;

  virtual vec2d<U> get_coords_range(const IndexRange& range) const;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_kernels_matrix_kernel_h
