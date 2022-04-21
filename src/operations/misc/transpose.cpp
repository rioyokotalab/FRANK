#include "FRANK/operations/misc.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/classes/matrix_proxy.h"
#include "FRANK/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>


namespace FRANK
{

declare_method(MatrixProxy, transpose_omm, (virtual_<const Matrix&>))

MatrixProxy transpose(const Matrix& A) { return transpose_omm(A); }

define_method(MatrixProxy, transpose_omm, (const Dense& A)) {
  Dense transposed(A.dim[1], A.dim[0]);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      transposed(j,i) = A(i,j);
    }
  }
  return transposed;
}

define_method(MatrixProxy, transpose_omm, (const LowRank& A)) {
  LowRank transposed(transpose(A.V), transpose(A.S), transpose(A.U));
  return transposed;
}

define_method(MatrixProxy, transpose_omm, (const Hierarchical& A)) {
  Hierarchical transposed(A.dim[1], A.dim[0]);
  for(int64_t i=0; i<A.dim[0]; i++) {
    for(int64_t j=0; j<A.dim[1]; j++) {
      transposed(j, i) = transpose(A(i, j));
    }
  }
  return transposed;
}

define_method(MatrixProxy, transpose_omm, (const Matrix& A)) {
  omm_error_handler("transpose", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
