#ifndef hicma_classes_initialization_helpers_matrix_kernels_arange_kernel_h
#define hicma_classes_initialization_helpers_matrix_kernels_arange_kernel_h

#include "hicma/classes/initialization_helpers/matrix_kernels/matrix_kernel.h"


namespace hicma
{

template<typename U = double>
class ArangeKernel : public MatrixKernel<U> {
  public:
    ArangeKernel() = default;
    ArangeKernel(const ArangeKernel& kernel) = default;
    ArangeKernel(ArangeKernel&& kernel) = default;
    ~ArangeKernel() = default;

    std::unique_ptr<MatrixKernel<U>> clone() const override;

    void apply(Matrix& A, int64_t row_start=0, int64_t col_start=0) const override;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_kernels_arange_kernel_h
