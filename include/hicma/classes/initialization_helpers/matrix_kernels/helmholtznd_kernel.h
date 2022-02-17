#ifndef hicma_classes_initialization_helpers_matrix_kernels_helmholtznd_kernel_h
#define hicma_classes_initialization_helpers_matrix_kernels_helmholtznd_kernel_h

#include "hicma/classes/initialization_helpers/matrix_kernels/parameterized_kernel.h"


namespace hicma
{

template<typename U = double>
class HelmholtzndKernel : public ParameterizedKernel<U> {
  public:
    HelmholtzndKernel(const vec2d<U>& params = vec2d<U>());
    HelmholtzndKernel(vec2d<U>&& params);
    HelmholtzndKernel(const HelmholtzndKernel& kernel) = default;
    HelmholtzndKernel(HelmholtzndKernel&& kernel) = default;
    ~HelmholtzndKernel() = default;

    std::unique_ptr<MatrixKernel<U>> clone() const override;

    std::unique_ptr<MatrixKernel<U>> move_clone() override;

    void apply(Matrix& A, int64_t row_start=0, int64_t col_start=0) const override;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_kernels_helmholtznd_kernel_h
