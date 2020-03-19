#include "hicma/classes/node.h"

#include "hicma/classes/index_range.h"

#include "yorel/yomm2/cute.hpp"

#include <memory>
#include <utility>


namespace hicma
{

std::unique_ptr<Node> Node::clone() const {
  return std::make_unique<Node>(*this);
}

std::unique_ptr<Node> Node::move_clone() {
  return std::make_unique<Node>(std::move(*this));
}

const char* Node::type() const { return "Node"; }

Node::Node(int i_abs, int j_abs, int level)
: i_abs(i_abs), j_abs(j_abs), level(level) {}

Node::Node(
  int i_abs, int j_abs, int level, IndexRange row_range, IndexRange col_range
) : i_abs(i_abs), j_abs(j_abs), level(level),
    row_range(row_range), col_range(col_range) {
}

bool Node::is_child(const Node& node) const {
  bool out = node.level >= level;
  out &= row_range.is_subrange(node.row_range);
  out &= col_range.is_subrange(node.col_range);
  return out;
}

} // namespace hicma
