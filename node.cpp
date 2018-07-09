#include <iostream>
#include "node.h"

namespace hicma {

  Node::~Node() {};

  Node* Node::clone() const {
    return new Node(*this);
  }

  const Node& Node::operator=(const double a) {
    return *this;
  }

  const Node& Node::operator=(const Node& A) {
    return *this;
  }

  const Node& Node::operator=(const NodePtr A) {
    return *this = *A;
  }

  const bool Node::is(const int enum_id) const {
    return enum_id == HICMA_NODE;
  }

  const char* Node::is_string() const { return "Node"; }

  NodePtr Node::add(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return NodePtr(nullptr);
  };

  NodePtr Node::sub(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return NodePtr(nullptr);
  };

  NodePtr Node::mul(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return NodePtr(nullptr);
  };

  double Node::norm() const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return 0.0;
  };

  void Node::print() const {};

  void Node::getrf() {};

  void Node::trsm(const Node& A, const char& uplo) {};

  void Node::gemm(const Node& A, const Node& B) {};

  NodePtr operator+(const Node& A, const Node& B) {
    return A.add(B);
  }

  NodePtr operator+(
      const Node& A,
      const NodePtr B) {
    return A.add(*B);
  }

  NodePtr operator+(
      const NodePtr A,
      const Node& B) {
    return A->add(B);
  }

  NodePtr operator+(
      const NodePtr A,
      const NodePtr B) {
    return A->add(*B);
  }

  const Node& operator+=(Node& A, const NodePtr B) {
    return A += *B;
  }

  const Node& operator+=(Node& A, const Node& B) {
    A = A.add(B);
    return A;
  }

  const NodePtr operator+=(NodePtr A, const Node& B) {
    A = (*A).add(B);
    return A;
  }

  NodePtr operator-(const Node& A, const Node& B) {
    return A.sub(B);
  }

  NodePtr operator-(
      const Node& A,
      const NodePtr B) {
    return A.sub(*B);
  }

  NodePtr operator-(
      const NodePtr A,
      const Node& B) {
    return A->sub(B);
  }

  NodePtr operator-(
      const NodePtr A,
      const NodePtr B) {
    return A->sub(*B);
  }

  const Node& operator-=(Node& A, const NodePtr B) {
    return A -= *B;
  }

  const Node& operator-=(Node& A, const Node& B) {
    A = A.sub(B);
    return A;
  }

  NodePtr operator*(const Node& A, const Node& B) {
    return A.mul(B);
  }

  NodePtr operator*(
      const Node& A,
      const NodePtr B) {
    return A.mul(*B);
  }

  NodePtr operator*(
      const NodePtr A,
      const Node& B) {
    return A->mul(B);
  }

  NodePtr operator*(
      const NodePtr A,
      const NodePtr B) {
    return A->mul(*B);
  }

  const Node& operator*=(Node& A, const NodePtr B) {
    return A *= *B;
  }

  const Node& operator*=(Node& A, const Node& B) {
    A = A.mul(B);
    return A;
  }

}
