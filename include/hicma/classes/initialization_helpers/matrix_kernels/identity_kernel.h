#ifndef hicma_classes_initialization_helpers_matrix_kernels_identity_kernel_h
#define hicma_classes_initialization_helpers_matrix_kernels_identity_kernel_h

#include "hicma/classes/initialization_helpers/matrix_kernels/matrix_kernel.h"


namespace hicma
{

template<typename U = double>
class IdentityKernel : public MatrixKernel<U> {
  public:
    IdentityKernel() = default;
    IdentityKernel(const IdentityKernel& kernel) = default;
    IdentityKernel(IdentityKernel&& kernel) = default;
    ~IdentityKernel() = default;

    std::unique_ptr<MatrixKernel<U>> clone() const override;

    std::unique_ptr<MatrixKernel<U>> move_clone() override;

    void apply(Matrix& A, int64_t row_start=0, int64_t col_start=0) const override;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_kernels_identity_kernel_h
