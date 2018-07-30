#include <iostream>
#include "node.h"

namespace hicma {

  _Node::~_Node() {};

  _Node* _Node::clone() const {
    return new _Node(*this);
  }

  const _Node& _Node::operator=(const double a) {
    return *this;
  }

  const _Node& _Node::operator=(const _Node& A) {
    return *this;
  }

  const _Node& _Node::operator=(const Node& A) {
    return *this = *A;
  }

  const bool _Node::is(const int enum_id) const {
    return enum_id == HICMA_NODE;
  }

  const char* _Node::is_string() const { return "_Node"; }

  Node _Node::add(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return Node(nullptr);
  };

  Node _Node::sub(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return Node(nullptr);
  };

  Node _Node::mul(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return Node(nullptr);
  };

  double _Node::norm() const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return 0.0;
  };

  void _Node::print() const {};

  void _Node::getrf() {};

  void _Node::trsm(const Node& A, const char& uplo) {};

  void _Node::gemm(const Node& A, const Node& B) {};

  Node operator+(const Node& A, const Node& B) {
    return A->add(B);
  }

  const Node operator+=(const Node A, const Node& B) {
    *A = A->add(B);
    return A;
  }

  Node operator-(const Node& A, const Node& B) {
    return A->sub(B);
  }

  Node operator-=(Node A, const Node& B) {
    *A = A->sub(B);
    return A;
  }

  Node operator*(const Node& A, const Node& B) {
    return A->mul(B);
  }

}
