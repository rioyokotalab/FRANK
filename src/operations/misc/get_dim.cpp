#include "FRANK/operations/misc.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/empty.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <cstdlib>


namespace FRANK
{

declare_method(int64_t, get_n_rows_omm, (virtual_<const Matrix&>))

int64_t get_n_rows(const Matrix& A) { return get_n_rows_omm(A); }

define_method(int64_t, get_n_rows_omm, (const Dense& A)) { return A.dim[0]; }

define_method(int64_t, get_n_rows_omm, (const Empty& A)) { return A.dim[0]; }

define_method(int64_t, get_n_rows_omm, (const LowRank& A)) { return A.dim[0]; }

define_method(int64_t, get_n_rows_omm, (const Hierarchical& A)) {
  int64_t n_rows = 0;
  for (int64_t i=0; i<A.dim[0]; i++) {
    n_rows += get_n_rows(A(i, 0));
  }
  return n_rows;
}

define_method(int64_t, get_n_rows_omm, (const Matrix& A)) {
  omm_error_handler("get_n_rows", {A}, __FILE__, __LINE__);
  std::abort();
}


declare_method(int64_t, get_n_cols_omm, (virtual_<const Matrix&>))

int64_t get_n_cols(const Matrix& A) { return get_n_cols_omm(A); }

define_method(int64_t, get_n_cols_omm, (const Dense& A)) { return A.dim[1]; }

define_method(int64_t, get_n_cols_omm, (const Empty& A)) { return A.dim[1]; }

define_method(int64_t, get_n_cols_omm, (const LowRank& A)) { return A.dim[1]; }

define_method(int64_t, get_n_cols_omm, (const Hierarchical& A)) {
  int64_t n_cols = 0;
  for (int64_t j=0; j<A.dim[1]; j++) {
    n_cols += get_n_cols(A(0, j));
  }
  return n_cols;
}

define_method(int64_t, get_n_cols_omm, (const Matrix& A)) {
  omm_error_handler("get_n_cols", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace FRANK
