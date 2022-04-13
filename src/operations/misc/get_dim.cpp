#include "hicma/operations/misc.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/empty.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <cstdint>
#include <cstdlib>


namespace hicma
{

int64_t get_n_rows(const Matrix& A) { return get_n_rows_omm(A); }

template<typename T>
int64_t get_n_rows_dense(const Dense<T>& A) { return A.dim[0]; }

define_method(int64_t, get_n_rows_omm, (const Dense<double>& A)) { return get_n_rows_dense(A); }

define_method(int64_t, get_n_rows_omm, (const Dense<float>& A)) { return get_n_rows_dense(A); }

define_method(int64_t, get_n_rows_omm, (const Empty& A)) { return A.dim[0]; }

template<typename T>
int64_t get_n_rows_low_rank(const LowRank<T>& A) { return A.dim[0]; }

define_method(int64_t, get_n_rows_omm, (const LowRank<double>& A)) { return get_n_rows_low_rank(A); }

define_method(int64_t, get_n_rows_omm, (const LowRank<float>& A)) { return get_n_rows_low_rank(A); }

template<typename T>
int64_t get_n_rows_hierarchical(const Hierarchical<T>& A) {
  int64_t n_rows = 0;
  for (int64_t i=0; i<A.dim[0]; i++) {
    n_rows += get_n_rows(A(i, 0));
  }
  return n_rows; 
}

define_method(int64_t, get_n_rows_omm, (const Hierarchical<double>& A)) {
  return get_n_rows_hierarchical(A);
}

define_method(int64_t, get_n_rows_omm, (const Hierarchical<float>& A)) {
  return get_n_rows_hierarchical(A);
}

define_method(int64_t, get_n_rows_omm, (const Matrix& A)) {
  omm_error_handler("get_n_rows", {A}, __FILE__, __LINE__);
  std::abort();
}


int64_t get_n_cols(const Matrix& A) { return get_n_cols_omm(A); }

template<typename T>
int64_t get_n_cols_dense(const Dense<T>& A) { return A.dim[1]; }

define_method(int64_t, get_n_cols_omm, (const Dense<double>& A)) { return get_n_cols_dense(A); }

define_method(int64_t, get_n_cols_omm, (const Dense<float>& A)) { return get_n_cols_dense(A); }

define_method(int64_t, get_n_cols_omm, (const Empty& A)) { return A.dim[1]; }

template<typename T>
int64_t get_n_cols_low_rank(const LowRank<T>& A) { return A.dim[1]; }

define_method(int64_t, get_n_cols_omm, (const LowRank<double>& A)) { return get_n_cols_low_rank(A); }

define_method(int64_t, get_n_cols_omm, (const LowRank<float>& A)) { return get_n_cols_low_rank(A); }

template<typename T>
int64_t get_n_cols_hierarchical(const Hierarchical<T>& A) {
  int64_t n_cols = 0;
  for (int64_t j=0; j<A.dim[1]; j++) {
    n_cols += get_n_cols(A(0, j));
  }
  return n_cols; 
}

define_method(int64_t, get_n_cols_omm, (const Hierarchical<double>& A)) {
  return get_n_cols_hierarchical(A);
}

define_method(int64_t, get_n_cols_omm, (const Hierarchical<float>& A)) {
  return get_n_cols_hierarchical(A);
}

define_method(int64_t, get_n_cols_omm, (const Matrix& A)) {
  omm_error_handler("get_n_cols", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
