#include "hicma/classes/matrix_proxy.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/no_copy_split.h"
#include "hicma/classes/uniform_hierarchical.h"
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

// TODO This might not play nice with NoCopySplit (pointer replacing doesn't cut
// it!). Consider MatrixProxyView?
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


define_method(std::unique_ptr<Matrix>, clone, (const Dense& A)) {
  return std::make_unique<Dense>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const LowRank& A)) {
  return std::make_unique<LowRank>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const LowRankShared& A)) {
  return std::make_unique<LowRankShared>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const Hierarchical& A)) {
  return std::make_unique<Hierarchical>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const UniformHierarchical& A)) {
  return std::make_unique<UniformHierarchical>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const NoCopySplit& A)) {
  return std::make_unique<NoCopySplit>(A);
}

define_method(std::unique_ptr<Matrix>, clone, (const Matrix& A)) {
  omm_error_handler("clone", {A}, __FILE__, __LINE__);
  std::abort();
}

define_method(std::unique_ptr<Matrix>, move_clone, (Dense&& A)) {
  return std::make_unique<Dense>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (LowRank&& A)) {
  return std::make_unique<LowRank>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (LowRankShared&& A)) {
  return std::make_unique<LowRankShared>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (Hierarchical&& A)) {
  return std::make_unique<Hierarchical>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (UniformHierarchical&& A)) {
  return std::make_unique<UniformHierarchical>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (NoCopySplit&& A)) {
  return std::make_unique<NoCopySplit>(std::move(A));
}

define_method(std::unique_ptr<Matrix>, move_clone, (Matrix&& A)) {
  omm_error_handler("move_clone", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
