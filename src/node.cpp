#include "hicma/node.h"

#include <iostream>

#include "yorel/multi_methods.hpp"

namespace hicma {

  Node::Node() : i_abs(0), j_abs(0), level(0) { MM_INIT(); }

  Node::Node(
             const int _i_abs,
             const int _j_abs,
             const int _level)
    : i_abs(_i_abs), j_abs(_j_abs), level(_level) { MM_INIT(); }

  Node::Node(const Node& A)
    : i_abs(A.i_abs), j_abs(A.j_abs), level(A.level) { MM_INIT(); }

  Node::~Node() {};

  Node* Node::clone() const {
    return new Node(*this);
  }

  const Node& Node::operator=(Node&& A) {
    return *this;
  }

  const char* Node::type() const { return "Node"; }

  double Node::norm() const {
    std::cerr << "No norm for Node type." << std::endl; abort();
    return 0.0;
  };

  void Node::print() const {};

  void Node::transpose() {};

  // void Node::geqrt(Dense& T) {};

  // void Node::geqrt(Hierarchical& T) {};

  // void Node::larfb(const Dense& Y, const Dense& T, const bool trans) {};

  // void Node::larfb(const Hierarchical& Y, const Hierarchical& T, const bool trans) {};

  // void Node::tpqrt(Dense& A, Dense& T) {};

  // void Node::tpqrt(Hierarchical& A, Dense& T) {};

  // void Node::tpqrt(Hierarchical& A, Hierarchical& T) {};

  // void Node::tpmqrt(Dense& B, const Dense& Y, const Dense& T, const bool trans) {};

  // void Node::tpmqrt(Dense& B, const LowRank& Y, const Dense& T, const bool trans) {};

  // void Node::tpmqrt(Dense& B, const Hierarchical& Y, const Hierarchical& T, const bool trans) {};

  // void Node::tpmqrt(LowRank& B, const Dense& Y, const Dense& T, const bool trans) {};

  // void Node::tpmqrt(LowRank& B, const LowRank& Y, const Dense& T, const bool trans) {};

  // void Node::tpmqrt(Hierarchical& B, const Dense& Y, const Dense& T, const bool trans) {};

  // void Node::tpmqrt(Hierarchical& B, const Hierarchical& Y, const Hierarchical& T, const bool trans) {};

}
