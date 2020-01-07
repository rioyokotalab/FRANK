#include "hicma/classes/node.h"

#include "hicma/classes/index_range.h"

#include <iostream>
#include <memory>
#include <utility>

#include "yorel/multi_methods.hpp"

namespace hicma {

  Node::Node() : i_abs(0), j_abs(0), level(0) { MM_INIT(); }

  Node::~Node() = default;

  Node::Node(const Node& A) {
    MM_INIT();
    *this = A;
  }

  Node& Node::operator=(const Node& A) = default;

  Node::Node(Node&& A) {
    MM_INIT();
    *this = std::move(A);
  }

  Node& Node::operator=(Node&& A) = default;

  std::unique_ptr<Node> Node::clone() const {
    return std::make_unique<Node>(*this);
  }

  std::unique_ptr<Node> Node::move_clone() {
    return std::make_unique<Node>(std::move(*this));
  }

  const char* Node::type() const { return "Node"; }

  Node::Node(int i_abs, int j_abs, int level)
  : i_abs(i_abs), j_abs(j_abs), level(level) { MM_INIT(); }

  Node::Node(
    int i_abs, int j_abs, int level, IndexRange row_range, IndexRange col_range
  ) : i_abs(i_abs), j_abs(j_abs), level(level),
      row_range(row_range), col_range(col_range) {
    MM_INIT();
  }

  bool Node::is_child(const Node& node) const {
    bool out = node.level == level + 1;
    out &= row_range.is_subrange(node.row_range);
    out &= col_range.is_subrange(node.col_range);
    return out;
  }

} // namespace hicma
