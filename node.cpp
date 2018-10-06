#include "hierarchical.h"

namespace hicma {

  Node::Node() : i_abs(0), j_abs(0), level(0) {}

  Node::Node(
             const int _i_abs,
             const int _j_abs,
             const int _level)
    : i_abs(_i_abs), j_abs(_j_abs), level(_level) {}

  Node::Node(const Node& ref)
    : i_abs(ref.i_abs), j_abs(ref.j_abs), level(ref.level) {}

  Node::~Node() {};

  Node* Node::clone() const {
    return new Node(*this);
  }

  void swap(Node& A, Node& B) {
    using std::swap;
    swap(A.i_abs, B.i_abs);
    swap(A.j_abs, B.j_abs);
    swap(A.level, B.level);
  }

  const Node& Node::operator=(Block A) {
    return *this = *A.ptr;
  }

  const bool Node::is(const int enum_id) const {
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

  void Node::gemm(const Node& A, const Node& B) {};
}
