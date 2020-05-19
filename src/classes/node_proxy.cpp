#include "hicma/classes/node_proxy.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/node.h"
#include "hicma/classes/no_copy_split.h"
#include "hicma/classes/uniform_hierarchical.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <memory>


namespace hicma
{

// Reconsider these constructors. Performance testing needed!!
NodeProxy::NodeProxy(const NodeProxy& A) : ptr(clone(A)) {}

// TODO This might not play nice with NoCopySplit (pointer replacing doesn't cut
// it!). Consider NodeProxyView?
NodeProxy& NodeProxy::operator=(const NodeProxy& A) {
  ptr = clone(A);
  return *this;
}

NodeProxy::NodeProxy(const Node& A) : ptr(clone(A)) {}

NodeProxy::NodeProxy(Node&& A) : ptr(move_clone(std::move(A))) {}

NodeProxy::operator const Node&() const {
  assert(ptr.get() != nullptr);
  return *ptr;
}

NodeProxy::operator Node&() {
  assert(ptr.get() != nullptr);
  return *ptr;
}

const char* NodeProxy::type() const { return ptr->type(); }

define_method(std::unique_ptr<Node>, clone, (const Dense& A)) {
  return std::make_unique<Dense>(A);
}

define_method(std::unique_ptr<Node>, clone, (const LowRank& A)) {
  return std::make_unique<LowRank>(A);
}

define_method(std::unique_ptr<Node>, clone, (const LowRankShared& A)) {
  return std::make_unique<LowRankShared>(A);
}

define_method(std::unique_ptr<Node>, clone, (const Hierarchical& A)) {
  return std::make_unique<Hierarchical>(A);
}

define_method(std::unique_ptr<Node>, clone, (const UniformHierarchical& A)) {
  return std::make_unique<UniformHierarchical>(A);
}

define_method(std::unique_ptr<Node>, clone, (const NoCopySplit& A)) {
  return std::make_unique<NoCopySplit>(A);
}

define_method(std::unique_ptr<Node>, clone, (const Node& A)) {
  omm_error_handler("clone", {A}, __FILE__, __LINE__);
  abort();
}

define_method(std::unique_ptr<Node>, move_clone, (Dense&& A)) {
  return std::make_unique<Dense>(std::move(A));
}

define_method(std::unique_ptr<Node>, move_clone, (LowRank&& A)) {
  return std::make_unique<LowRank>(std::move(A));
}

define_method(std::unique_ptr<Node>, move_clone, (LowRankShared&& A)) {
  return std::make_unique<LowRankShared>(std::move(A));
}

define_method(std::unique_ptr<Node>, move_clone, (Hierarchical&& A)) {
  return std::make_unique<Hierarchical>(std::move(A));
}

define_method(std::unique_ptr<Node>, move_clone, (UniformHierarchical&& A)) {
  return std::make_unique<UniformHierarchical>(std::move(A));
}

define_method(std::unique_ptr<Node>, move_clone, (NoCopySplit&& A)) {
  return std::make_unique<NoCopySplit>(std::move(A));
}

define_method(std::unique_ptr<Node>, move_clone, (Node&& A)) {
  omm_error_handler("move_clone", {A}, __FILE__, __LINE__);
  abort();
}

} // namespace hicma
