#include "node.h"

#include <iostream>
#include <vector>

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

  void swap(Node& first, Node& second) {
    using std::swap;
    swap(first.i_abs, second.i_abs);
    swap(first.j_abs, second.j_abs);
    swap(first.level, second.level);
  }

  const Node& Node::operator=(const double a) {
    return *this;
  }

  const Node& Node::operator=(const Node& A) {
    return *this;
  }
  const Node& Node::operator=(Node&& A) {
    return *this;
  }

  const Node& Node::operator=(Block A) {
    return *this = *A.ptr;
  }

  Block Node::operator+(const Node& A) const {
    std::cerr << "Operations should not be performed between two node types." << std::endl;
    abort();
    return Block();
  };
  Block Node::operator+(Block&& A) const {
    std::cerr << "Operations should not be performed between two node types." << std::endl;
    abort();
    return Block();
  };
  const Node& Node::operator+=(const Node& A) {
    std::cerr << "Operations should not be performed between two node types." << std::endl;
    abort();
    return *this;
  };
  const Node& Node::operator+=(Block&& A) {
    std::cerr << "Operations should not be performed between two node types." << std::endl;
    abort();
    return *this;
  };
  Block Node::operator-(const Node& A) const {
    std::cerr << "Operations should not be performed between two node types." << std::endl;
    abort();
    return Block();
  };
  Block Node::operator-(Block&& A) const {
    std::cerr << "Operations should not be performed between two node types." << std::endl;
    abort();
    return Block();
  };
  const Node& Node::operator-=(const Node& A) {
    std::cerr << "Operations should not be performed between two node types." << std::endl;
    abort();
    return *this;
  };
  const Node& Node::operator-=(Block&& A) {
    std::cerr << "Operations should not be performed between two node types." << std::endl;
    abort();
    return *this;
  };
  Block Node::operator*(const Node& A) const {
    std::cerr << "Operations should not be performed between two node types." << std::endl;
    abort();
    return Block();
  };
  Block Node::operator*(Block&& A) const {
    std::cerr << "Operations should not be performed between two node types." << std::endl;
    abort();
    return Block();
  };

  const bool Node::is(const int enum_id) const {
    return enum_id == HICMA_NODE;
  }

  const char* Node::is_string() const { return "Node"; }

  double Node::norm() const {
    std::cerr << "No norm for Node type." << std::endl; abort();
    return 0.0;
  };

  void Node::print() const {};

  void Node::getrf() {};

  void Node::trsm(const Node& A, const char& uplo) {};

  void Node::gemm(const Node& A, const Node& B) {};
}
