#include "hicma/operations/arithmetic.h"
#include "hicma/extension_headers/operations.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/util/omm_error_handler.h"

#include <cstdint>
#include <cstdlib>


namespace hicma
{

Matrix& operator*=(Matrix& A, double b) {
  return multiplication_omm(A, b);
}

template<typename T>
Dense<T>& multiply_dense(Dense<T>& A, double b) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) *= b;
    }
  }
  return A;
}

define_method(
  Matrix&, multiplication_omm, (Dense<float>& A, double b)
) {
  return multiply_dense(A, b);
}

define_method(
  Matrix&, multiplication_omm, (Dense<double>& A, double b)
) {
  return multiply_dense(A, b);
}

template<typename T>
LowRank<T>& multiply_low_rank(LowRank<T>& A, double b) {
  A.S *= b;
  return A;
}

define_method(
  Matrix&, multiplication_omm, (LowRank<float>& A, double b)
) {
  return multiply_low_rank(A, b);
}

define_method(
  Matrix&, multiplication_omm, (LowRank<double>& A, double b)
) {
  return multiply_low_rank(A, b);
}

template<typename T>
Hierarchical<T>& multiply_hierarchical(Hierarchical<T>& A, double b) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) *= b;
    }
  }
  return A;
}

define_method(
  Matrix&, multiplication_omm, (Hierarchical<float>& A, double b)
) {
  return multiply_hierarchical(A, b);
}

define_method(
  Matrix&, multiplication_omm, (Hierarchical<double>& A, double b)
) {
  return multiply_hierarchical(A, b);
}

define_method(Matrix&, multiplication_omm, (Matrix& A, double)) {
  omm_error_handler("operator*<double>", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
