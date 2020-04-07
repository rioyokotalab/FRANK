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

} // namespace hicma
