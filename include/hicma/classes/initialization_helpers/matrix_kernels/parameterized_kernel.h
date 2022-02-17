#ifndef hicma_classes_initialization_helpers_matrix_kernels_parameterized_kernel_h
#define hicma_classes_initialization_helpers_matrix_kernels_parameterized_kernel_h

#include "hicma/classes/initialization_helpers/matrix_kernels/matrix_kernel.h"


namespace hicma
{

template<typename U = double>
class ParameterizedKernel : public MatrixKernel<U>{
  protected:
    vec2d<U> params;

  public:
    ParameterizedKernel(const vec2d<U>& params = vec2d<U>());
    ParameterizedKernel(vec2d<U>&& params);
    ParameterizedKernel(const ParameterizedKernel& kernel) = default;
    ParameterizedKernel(ParameterizedKernel&& kernel) = default;
    ~ParameterizedKernel() = default;

    virtual std::unique_ptr<MatrixKernel<U>> clone() const = 0;

    virtual std::unique_ptr<MatrixKernel<U>> move_clone() = 0;

    virtual void apply(Matrix& A, int64_t row_start=0, int64_t col_start=0) const = 0;

    vec2d<U> get_coords_range(const IndexRange& range) const override;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_kernels_parameterized_kernel_h
