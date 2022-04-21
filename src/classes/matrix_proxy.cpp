#include "FRANK/classes/matrix_proxy.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/empty.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdlib>
#include <memory>
#include <utility>


namespace FRANK
{

declare_method(std::unique_ptr<Matrix>, clone, (virtual_<const Matrix&>))

define_method(std::unique_ptr<Matrix>, clone, (const Dense& A)) {
  return std::make_unique<Dense>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const Empty& A)) {
  return std::make_unique<Empty>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const LowRank& A)) {
  return std::make_unique<LowRank>(A);
}
define_method(std::unique_ptr<Matrix>, clone, (const Hierarchical& A)) {
  return std::make_unique<Hierarchical>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const Matrix& A)) {
  omm_error_handler("clone", {A}, __FILE__, __LINE__);
  std::abort();
}

declare_method(std::unique_ptr<Matrix>, move_clone, (virtual_<Matrix&&>))

define_method(std::unique_ptr<Matrix>, move_clone, (Dense&& A)) {
  return std::make_unique<Dense>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (Empty&& A)) {
  return std::make_unique<Empty>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (LowRank&& A)) {
  return std::make_unique<LowRank>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (Hierarchical&& A)) {
  return std::make_unique<Hierarchical>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (Matrix&& A)) {
  omm_error_handler("move_clone", {A}, __FILE__, __LINE__);
  std::abort();
}

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

} // namespace FRANK
