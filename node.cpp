#include <iostream>
#include <memory>
#include <node.h>

namespace hicma {

  Node::~Node() {};

  Node* Node::clone() const {
    return new Node(*this);
  }

  const Node& Node::assign(const double a) {
    return *this;
  }

  const Node& Node::operator=(const Node& A) {
    return *this;
  }

  const Node& Node::operator=(const std::shared_ptr<Node> A) {
    return *this = *A;
  }

  const bool Node::is(const int enum_id) const {
    return enum_id == HICMA_NODE;
  }

  const char* Node::is_string() const { return "Node"; }

  std::shared_ptr<Node> Node::add(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return std::shared_ptr<Node>(nullptr);
  };

  std::shared_ptr<Node> Node::sub(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return std::shared_ptr<Node>(nullptr);
  };

  std::shared_ptr<Node> Node::mul(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return std::shared_ptr<Node>(nullptr);
  };

  double Node::norm_test() const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return 0.0;
  };

  void Node::getrf_test() {};

  void Node::trsm(const Node& A, const char& uplo) {};

  void Node::gemm(const Node& A, const Node& B) {};

  std::shared_ptr<Node> operator+(const Node& A, const Node& B) {
    return A.add(B);
  }

  const Node& operator+=(Node& A, const std::shared_ptr<Node> B) {
    return A += *B;
  }

  const Node& operator+=(Node& A, const Node& B) {
    A = A.add(B);
    return A;
  }

  std::shared_ptr<Node> operator-(const Node& A, const Node& B) {
    return A.sub(B);
  }

  const Node& operator-=(Node& A, const std::shared_ptr<Node> B) {
    return A -= *B;
  }

  const Node& operator-=(Node& A, const Node& B) {
    A = A.sub(B);
    return A;
  }

  std::shared_ptr<Node> operator*(const Node& A, const Node& B) {
    return A.mul(B);
  }

  const Node& operator*=(Node& A, const std::shared_ptr<Node> B) {
    return A *= *B;
  }

  const Node& operator*=(Node& A, const Node& B) {
    A = A.mul(B);
    return A;
  }

}
