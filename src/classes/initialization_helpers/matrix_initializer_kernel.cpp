#include "hicma/classes/initialization_helpers/matrix_initializer_kernel.h"

#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <iostream>


namespace hicma
{

MatrixInitializerKernel::MatrixInitializerKernel(
  std::function<double(uint64_t, uint64_t, const vec2d<double>&, uint64_t)> kernel, double admis, 
  int64_t rank, int admis_type) : 
  MatrixInitializer<MatrixInitializerKernel>(admis, rank, admis_type) {
    kernel = MatrixKernel(kernel);
  }

void MatrixInitializerKernel<U>::fill_dense_representation(
  Matrix& A, const IndexRange& row_range, const IndexRange& col_range, const vec2d<double>& params
) const {
  try {
    kernel.apply(dynamic_cast<Dense<double>&>(A), params.row_range.start, col_range.start);
  }
  catch (std::bad_cast& e) {
    try {
      kernel.apply(dynamic_cast<Dense<float>&>(A), params.row_range.start, col_range.start);
    }
    catch(std::bad_cast& e){
      std::cerr<<"Can not fill a non-dense matrix"<<std::endl;
    }
  }
}


} // namespace hicma
