#include "hicma/classes/initialization_helpers/matrix_kernels/parameterized_kernel.h"

#include "hicma/classes/initialization_helpers/index_range.h"

#include <utility>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class ParameterizedKernel<float>;
template class ParameterizedKernel<double>;

template<typename U>
ParameterizedKernel<U>::ParameterizedKernel(const vec2d<U>& params) :
                        params(params) {}

template<typename U>
ParameterizedKernel<U>::ParameterizedKernel(vec2d<U>&& params) :
                        params(std::move(params)) {}

template<typename U>
vec2d<U> ParameterizedKernel<U>::get_coords_range(const IndexRange& range) const {
  vec2d<U> coords_range;
  for(size_t d=0; d<params.size(); d++)
    coords_range.push_back(std::vector<U>(params[d].begin()+range.start, params[d].begin()+range.start+range.n));
  return coords_range;
}

} // namespace hicma
