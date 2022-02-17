#ifndef hicma_classes_initialization_helpers_matrix_kernels_random_uniform_kernel_h
#define hicma_classes_initialization_helpers_matrix_kernels_random_uniform_kernel_h

#include "hicma/classes/initialization_helpers/matrix_kernels/matrix_kernel.h"

#include <random>


namespace hicma
{

template<typename U = double>
class RandomUniformKernel : public MatrixKernel<U> {
  private:
    uint32_t seed;
    static std::random_device rd;
    static std::mt19937 gen;
    static std::uniform_real_distribution<U> dist;

  public:
    RandomUniformKernel(uint32_t seed=0);
    RandomUniformKernel(const RandomUniformKernel& kernel) = default;
    RandomUniformKernel(RandomUniformKernel&& kernel) = default;
    ~RandomUniformKernel() = default;

    std::unique_ptr<MatrixKernel<U>> clone() const override;

    std::unique_ptr<MatrixKernel<U>> move_clone() override;

    void apply(Matrix& A, int64_t row_start=0, int64_t col_start=0) const override;

    void apply(Matrix& A, bool deterministic_seed, int64_t row_start=0, int64_t col_start=0) const;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_kernels_random_uniform_kernel_h
