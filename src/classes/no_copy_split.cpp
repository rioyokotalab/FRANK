#include "hicma/classes/no_copy_split.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/dense_view.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_view.h"
#include "hicma/classes/hierarchical.h"

#include "yorel/yomm2/cute.hpp"

#include <iostream>
#include <memory>
#include <utility>

namespace hicma
{

NoCopySplit::NoCopySplit() = default;

NoCopySplit::~NoCopySplit() = default;

NoCopySplit::NoCopySplit(const NoCopySplit& A) = default;

NoCopySplit& NoCopySplit::operator=(const NoCopySplit& A) = default;

NoCopySplit::NoCopySplit(NoCopySplit&& A) = default;

NoCopySplit& NoCopySplit::operator=(NoCopySplit&& A) = default;

std::unique_ptr<Node> NoCopySplit::clone() const {
  return std::make_unique<NoCopySplit>(*this);
}

std::unique_ptr<Node> NoCopySplit::move_clone() {
  return std::make_unique<NoCopySplit>(std::move(*this));
}

const char* NoCopySplit::type() const {
  return "NoCopySplit";
}

NoCopySplit::NoCopySplit(
  Node& A, int ni_level, int nj_level, bool node_only
) : Hierarchical(A, ni_level, nj_level, true) {
  if (!node_only) {
    *this = make_no_copy_split(A, ni_level, nj_level);
  }
}

define_method(
  NoCopySplit, make_no_copy_split,
  (Dense& A, int ni_level, int nj_level)
) {
  NoCopySplit out(A, ni_level, nj_level, true);
  out.create_children();
  for (NodeProxy& child : out) {
    child = DenseView(child, A);
  }
  return out;
}

define_method(
  NoCopySplit, make_no_copy_split,
  (LowRank& A, int ni_level, int nj_level)
) {
  NoCopySplit out(A, ni_level, nj_level, true);
  out.create_children();
  for (NodeProxy& child : out) {
    child = LowRankView(child, A);
  }
  return out;
}

define_method(
  NoCopySplit, make_no_copy_split,
  (Node& A, [[maybe_unused]] int ni_level, [[maybe_unused]] int nj_level)
) {
  std::cout << "Cannot create NoCopySplit from " << A.type() << "!" << std::endl;
  abort();
}

NoCopySplit::NoCopySplit(
  const Node& A, int ni_level, int nj_level, bool node_only
) : Hierarchical(A, ni_level, nj_level, true) {
  if (!node_only) {
    *this = make_no_copy_split_const(A, ni_level, nj_level);
  }
}

define_method(
  NoCopySplit, make_no_copy_split_const,
  (const Dense& A, int ni_level, int nj_level)
) {
  NoCopySplit out(A, ni_level, nj_level, true);
  out.create_children();
  for (NodeProxy& child : out) {
    child = DenseView(child, A);
  }
  return out;
}

define_method(
  NoCopySplit, make_no_copy_split_const,
  (const LowRank& A, int ni_level, int nj_level)
) {
  NoCopySplit out(A, ni_level, nj_level, true);
  out.create_children();
  for (NodeProxy& child : out) {
    child = LowRankView(child, A);
  }
  return out;
}

define_method(
  NoCopySplit, make_no_copy_split_const,
  (const Node& A, [[maybe_unused]] int ni_level, [[maybe_unused]] int nj_level)
) {
  std::cout << "Cannot create NoCopySplit from " << A.type() << "!" << std::endl;
  abort();
}

} // namespace hicma
