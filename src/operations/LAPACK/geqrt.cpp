#include "hicma/operations/LAPACK.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/matrix.h"
#include "hicma/util/omm_error_handler.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>


namespace hicma
{

void geqrt(Matrix& A, Matrix& T) { geqrt_omm(A, T); }

// single precision
define_method(void, geqrt_omm, (Dense<float>& A, Dense<float>& T)) {
  assert(T.dim[0] == A.dim[1]);
  assert(T.dim[1] == A.dim[1]);
  LAPACKE_sgeqrt3(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &T, T.stride
  );
}

// double precision
define_method(void, geqrt_omm, (Dense<double>& A, Dense<double>& T)) {
  assert(T.dim[0] == A.dim[1]);
  assert(T.dim[1] == A.dim[1]);
  LAPACKE_dgeqrt3(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &T, T.stride
  );
}

template<typename T>
void hierarchical_geqrt(Hierarchical<T>& A, Hierarchical<T>& B){
  std::cerr << "Possibly not fully implemented yet. Read code!!!" << std::endl;
  for(int64_t k = 0; k < A.dim[1]; k++) {
    geqrt(A(k, k), B(k, k));
    for(int64_t j = k+1; j < A.dim[1]; j++) {
      larfb(A(k, k), B(k, k), A(k, j), true);
    }
    for(int64_t i = k+1; i < A.dim[0]; i++) {
      tpqrt(A(k, k), A(i, k), B(i, k));
      for(int64_t j = k+1; j < A.dim[1]; j++) {
        tpmqrt(A(i, k), B(i, k), A(k, j), A(i, j), true);
      }
    }
  }
}

define_method(void, geqrt_omm, (Hierarchical<float>& A, Hierarchical<float>& T)) {
  hierarchical_geqrt(A, T);
}

define_method(void, geqrt_omm, (Hierarchical<double>& A, Hierarchical<double>& T)) {
  hierarchical_geqrt(A, T);
}

// Fallback default, abort with error message
define_method(void, geqrt_omm, (Matrix& A, Matrix& T)) {
  omm_error_handler("geqrt", {A, T}, __FILE__, __LINE__);
  std::abort();
}

template<>
void geqrt2(Dense<float>& A, Dense<float>& T) {
  assert(T.dim[0] == A.dim[1]);
  assert(T.dim[1] == A.dim[1]);
  LAPACKE_sgeqrt2(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &T, T.stride
  );
}

template<>
void geqrt2(Dense<double>& A, Dense<double>& T) {
  assert(T.dim[0] == A.dim[1]);
  assert(T.dim[1] == A.dim[1]);
  LAPACKE_dgeqrt2(
    LAPACK_ROW_MAJOR,
    A.dim[0], A.dim[1],
    &A, A.stride,
    &T, T.stride
  );
}

} // namespace hicma
