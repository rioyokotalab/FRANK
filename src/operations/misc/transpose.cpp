#include "hicma/operations/misc.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <cstdint>


namespace hicma
{

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

} // namespace hicma
