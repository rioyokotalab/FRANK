#include "hicma/classes/node.h"

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

  Node::Node(const int i_abs, const int j_abs, const int level)
  : i_abs(i_abs), j_abs(j_abs), level(level) { MM_INIT(); }

  std::unique_ptr<Node> Node::clone() const {
    return std::make_unique<Node>(*this);
  }

  std::unique_ptr<Node> Node::move_clone() {
    return std::make_unique<Node>(std::move(*this));
  }

  const char* Node::type() const { return "Node"; }

} // namespace hicma
