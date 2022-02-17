#ifndef hicma_classes_initialization_helpers_matrix_kernels_cauchy2d_kernel_h
#define hicma_classes_initialization_helpers_matrix_kernels_cauchy2d_kernel_h

#include "hicma/classes/initialization_helpers/matrix_kernels/parameterized_kernel.h"


namespace hicma
{

template<typename U = double>
class Cauchy2dKernel : public ParameterizedKernel<U> {
  public:
    Cauchy2dKernel(const vec2d<U>& params = vec2d<U>());
    Cauchy2dKernel(vec2d<U>&& params);
    Cauchy2dKernel(const Cauchy2dKernel& kernel) = default;
    Cauchy2dKernel(Cauchy2dKernel&& kernel) = default;
    ~Cauchy2dKernel() = default;

    std::unique_ptr<MatrixKernel<U>> clone() const override;

    void apply(Matrix& A, int64_t row_start=0, int64_t col_start=0) const override;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_kernels_cauchy2d_kernel_h
