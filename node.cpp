#include "hierarchical.h"

namespace hicma {

  Node::Node() : i_abs(0), j_abs(0), level(0) {}

  Node::Node(
             const int _i_abs,
             const int _j_abs,
             const int _level)
    : i_abs(_i_abs), j_abs(_j_abs), level(_level) {}

  Node::Node(const Node& A)
    : i_abs(A.i_abs), j_abs(A.j_abs), level(A.level) {}

  Node::~Node() {};

  Node* Node::clone() const {
    return new Node(*this);
  }

  const Node& Node::operator=(Node&& A) {
    return *this;
  }

  bool Node::is(const int enum_id) const {
    return enum_id == HICMA_NODE;
  }

  const char* Node::type() const { return "Node"; }

  double Node::norm() const {
    std::cerr << "No norm for Node type." << std::endl; abort();
    return 0.0;
  };

  void Node::print() const {};

  void Node::getrf() {};

  void Node::trsm(const Node& A, const char& uplo) {};

  void Node::gemm(const Node& A, const Node& B, const int& alpha, const int& beta) {};
}
