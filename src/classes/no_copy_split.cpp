#include "hicma/classes/no_copy_split.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>


namespace hicma
{

std::unique_ptr<Node> NoCopySplit::clone() const {
  return std::make_unique<NoCopySplit>(*this);
}

std::unique_ptr<Node> NoCopySplit::move_clone() {
  return std::make_unique<NoCopySplit>(std::move(*this));
}

const char* NoCopySplit::type() const { return "NoCopySplit"; }

NoCopySplit::NoCopySplit(Node& A, int64_t n_row_blocks, int64_t n_col_blocks)
: Hierarchical(n_row_blocks, n_col_blocks) {
  ClusterTree node(get_n_rows(A), get_n_cols(A), dim[0], dim[1]);
  for (ClusterTree& child_node : node) {
    (*this)[child_node] = make_view(child_node, A);
  }
}

NoCopySplit::NoCopySplit(Node& A, const Hierarchical& like)
: Hierarchical(like.dim[0], like.dim[1]) {
  assert(get_n_rows(A) == get_n_rows(like));
  assert(get_n_cols(A) == get_n_cols(like));
  ClusterTree node(like);
  for (ClusterTree& child_node : node) {
    (*this)[child_node] = make_view(child_node, A);
  }
}

define_method(NodeProxy, make_view, (const ClusterTree& node, Dense& A)) {
  return Dense(node.dim[0], node.dim[1], node.start[0], node.start[1], A);
}

define_method(
  NodeProxy, make_view, ([[maybe_unused]] const ClusterTree&, Node& A)
) {
  omm_error_handler("make_view", {A}, __FILE__, __LINE__);
  abort();
}

NoCopySplit::NoCopySplit(
  const Node& A, int64_t n_row_blocks, int64_t n_col_blocks
) : Hierarchical(n_row_blocks, n_col_blocks) {
  ClusterTree node(get_n_rows(A), get_n_cols(A), dim[0], dim[1]);
  for (ClusterTree& child_node : node) {
    (*this)[child_node] = make_view(child_node, A);
  }
}

NoCopySplit::NoCopySplit(const Node& A, const Hierarchical& like)
: Hierarchical(like.dim[0], like.dim[1]) {
  assert(get_n_rows(A) == get_n_rows(like));
  assert(get_n_cols(A) == get_n_cols(like));
  ClusterTree node(like);
  for (ClusterTree& child_node : node) {
    (*this)[child_node] = make_view(child_node, A);
  }
}

define_method(NodeProxy, make_view, (const ClusterTree& node, const Dense& A)) {
  return Dense(node.dim[0], node.dim[1], node.start[0], node.start[1], A);
}

define_method(
  NodeProxy, make_view, (const ClusterTree& node, const LowRank& A)
) {
  return LowRank(node.dim[0], node.dim[1], node.start[0], node.start[1], A);
}

define_method(
  NodeProxy, make_view, ([[maybe_unused]] const ClusterTree&, const Node& A)
) {
  omm_error_handler("make_view (const)", {A}, __FILE__, __LINE__);
  abort();
}

} // namespace hicma
