#include <iostream>
#include <memory>
#include <node.h>

namespace hicma {

  Node::~Node() {};

  const Node& Node::operator=(std::unique_ptr<Node> A) {
    return *this;
  }

  const bool Node::is(const int enum_id) const {
    return enum_id == HICMA_NODE;
  }

  const char* Node::is_string() const { return "Node"; }

  std::unique_ptr<Node> Node::add(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl;
    return std::unique_ptr<Node>(nullptr);
  };

  std::unique_ptr<Node> Node::sub(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl;
    return std::unique_ptr<Node>(nullptr);
  };

  std::unique_ptr<Node> Node::mul(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl;
    return std::unique_ptr<Node>(nullptr);
  };

  void Node::getrf_test() {};

  void Node::trsm(const Node& A, const char& uplo) {};

  void Node::gemm(const Node& A, const Node& B) {};

  std::unique_ptr<Node> operator+(const Node& A, const Node& B) {
    return A.add(B);
  }

  const Node operator+=(Node& A, const Node& B) {
    A = A + B;
    return A;
  }

  std::unique_ptr<Node> operator-(const Node& A, const Node& B) {
    return A.sub(B);
  }

  const Node operator-=(Node& A, const Node& B) {
    A = A - B;
    return A;
  }

  std::unique_ptr<Node> operator*(const Node& A, const Node& B) {
    return A.mul(B);
  }

  const Node operator*=(Node& A, const Node& B) {
    A = A * B;
    return A;
  }

}
