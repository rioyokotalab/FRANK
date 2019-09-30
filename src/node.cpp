#include "hicma/node.h"

#include <iostream>

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

  void Node::transpose() {};

  void Node::getrf() {};

  void Node::trsm(const Dense& A, const char& uplo) {};

  void Node::trsm(const Hierarchical& A, const char& uplo) {};

  void Node::gemm(const Dense& A, const Dense& B, const double& alpha, const double& beta) {};

  void Node::gemm(const Dense& A, const LowRank& B, const double& alpha, const double& beta) {};

  void Node::gemm(const Dense& A, const Hierarchical& B, const double& alpha, const double& beta) {};

  void Node::gemm(const LowRank& A, const Dense& B, const double& alpha, const double& beta) {};

  void Node::gemm(const LowRank& A, const LowRank& B, const double& alpha, const double& beta) {};

  void Node::gemm(const LowRank& A, const Hierarchical& B, const double& alpha, const double& beta) {};

  void Node::gemm(const Hierarchical& A, const Dense& B, const double& alpha, const double& beta) {};

  void Node::gemm(const Hierarchical& A, const LowRank& B, const double& alpha, const double& beta) {};

  void Node::gemm(const Hierarchical& A, const Hierarchical& B, const double& alpha, const double& beta) {};

  void Node::geqrt(Dense& T) {};

  void Node::geqrt(Hierarchical& T) {};

  void Node::larfb(const Dense& Y, const Dense& T, const bool trans) {};

  void Node::larfb(const Hierarchical& Y, const Hierarchical& T, const bool trans) {};

  void Node::tpqrt(Dense& A, Dense& T) {};

  void Node::tpqrt(Hierarchical& A, Dense& T) {};

  void Node::tpqrt(Hierarchical& A, Hierarchical& T) {};

  void Node::tpmqrt(Dense& B, const Dense& Y, const Dense& T, const bool trans) {};

  void Node::tpmqrt(Dense& B, const LowRank& Y, const Dense& T, const bool trans) {};

  void Node::tpmqrt(Dense& B, const Hierarchical& Y, const Hierarchical& T, const bool trans) {};

  void Node::tpmqrt(LowRank& B, const Dense& Y, const Dense& T, const bool trans) {};

  void Node::tpmqrt(LowRank& B, const LowRank& Y, const Dense& T, const bool trans) {};

  void Node::tpmqrt(Hierarchical& B, const Dense& Y, const Dense& T, const bool trans) {};

  void Node::tpmqrt(Hierarchical& B, const Hierarchical& Y, const Hierarchical& T, const bool trans) {};

}