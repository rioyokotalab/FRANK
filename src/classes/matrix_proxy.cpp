#include "hicma/classes/matrix_proxy.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/empty.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <cstdlib>
#include <memory>
#include <utility>


namespace hicma
{

// Reconsider these constructors. Performance testing needed!!
MatrixProxy::MatrixProxy(const MatrixProxy& A) : ptr(clone(A)) {}
MatrixProxy& MatrixProxy::operator=(const MatrixProxy& A) {
  ptr = clone(A);
  return *this;
}

MatrixProxy::MatrixProxy(const Matrix& A) : ptr(clone(A)) {}

MatrixProxy::MatrixProxy(Matrix&& A) : ptr(move_clone(std::move(A))) {}

MatrixProxy::operator const Matrix&() const {
  assert(ptr.get() != nullptr);
  return *ptr;
}

MatrixProxy::operator Matrix&() {
  assert(ptr.get() != nullptr);
  return *ptr;
}

define_method(std::unique_ptr<Matrix>, clone, (const Dense<float>& A)) {
  return std::make_unique<Dense<float>>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const Dense<double>& A)) {
  return std::make_unique<Dense<double>>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const Empty& A)) {
  return std::make_unique<Empty>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const LowRank<float>& A)) {
  return std::make_unique<LowRank<float>>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const LowRank<double>& A)) {
  return std::make_unique<LowRank<double>>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const Hierarchical<float>& A)) {
  return std::make_unique<Hierarchical<float>>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const Hierarchical<double>& A)) {
  return std::make_unique<Hierarchical<double>>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const Matrix& A)) {
  omm_error_handler("clone", {A}, __FILE__, __LINE__);
  std::abort();
}

define_method(std::unique_ptr<Matrix>, move_clone, (Dense<float>&& A)) {
  return std::make_unique<Dense<float>>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (Dense<double>&& A)) {
  return std::make_unique<Dense<double>>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (Empty&& A)) {
  return std::make_unique<Empty>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (LowRank<float>&& A)) {
  return std::make_unique<LowRank<float>>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (LowRank<double>&& A)) {
  return std::make_unique<LowRank<double>>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (Hierarchical<float>&& A)) {
  return std::make_unique<Hierarchical<float>>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (Hierarchical<double>&& A)) {
  return std::make_unique<Hierarchical<double>>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (Matrix&& A)) {
  omm_error_handler("move_clone", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
