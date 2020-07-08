#include "hicma/operations/misc.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <cstdint>
#include <cstdlib>


namespace hicma
{

int64_t get_n_rows(const Matrix& A) { return get_n_rows_omm(A); }

define_method(int64_t, get_n_rows_omm, (const Dense& A)) { return A.dim[0]; }

define_method(int64_t, get_n_rows_omm, (const LowRank& A)) { return A.dim[0]; }

define_method(int64_t, get_n_rows_omm, (const Hierarchical& A)) {
  int64_t n_rows = 0;
  for (int64_t i=0; i<A.dim[0]; i++) {
    n_rows += get_n_rows(A(i, 0));
  }
  return n_rows;
}

define_method(int64_t, get_n_rows_omm, (const SharedBasis& A)) {
  // TODO This can be made simpler if we store Vt instead of V
  int64_t out = 0;
  if (A.col_basis && A.num_child_basis() > 0) {
    for (int64_t i=0; i<A.num_child_basis(); ++i) {
      out += get_n_rows(A[i]);
    }
  } else {
    out = get_n_rows(*A.get_ptr());
  }
  return out;
}

define_method(int64_t, get_n_rows_omm, (const Matrix& A)) {
  omm_error_handler("get_n_rows", {A}, __FILE__, __LINE__);
  std::abort();
}


int64_t get_n_cols(const Matrix& A) { return get_n_cols_omm(A); }

define_method(int64_t, get_n_cols_omm, (const Dense& A)) { return A.dim[1]; }

define_method(int64_t, get_n_cols_omm, (const LowRank& A)) { return A.dim[1]; }

define_method(int64_t, get_n_cols_omm, (const Hierarchical& A)) {
  int64_t n_cols = 0;
  for (int64_t j=0; j<A.dim[1]; j++) {
    n_cols += get_n_cols(A(0, j));
  }
  return n_cols;
}

define_method(int64_t, get_n_cols_omm, (const SharedBasis& A)) {
  // TODO This can be made simpler if we store Vt instead of V
  int64_t out = 0;
  if (!A.col_basis && A.num_child_basis() > 0) {
    for (int64_t j=0; j<A.num_child_basis(); ++j) {
      out += get_n_cols(A[j]);
    }
  } else {
    out = get_n_cols(*A.get_ptr());
  }
  return out;
}

define_method(int64_t, get_n_cols_omm, (const Matrix& A)) {
  omm_error_handler("get_n_cols", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
