#include "hicma/classes/initialization_helpers/matrix_kernels/matrix_kernel.h"


namespace hicma
{

// explicit template initialization (these are the only available types)
template class MatrixKernel<float>;
template class MatrixKernel<double>;

template<typename U>
vec2d<U> MatrixKernel<U>::get_coords_range(const IndexRange&) const {
  return vec2d<U>();
}

} // namespace hicma
