#include <iostream>
#include "node.h"

namespace hicma {
  Node::Node() : i_abs(0), j_abs(0), level(0) {}
  Node::Node(
      const int _i_abs,
      const int _j_abs,
      const int _level) : i_abs(_i_abs), j_abs(_j_abs), level(_level) {}

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

  Block Node::operator+(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return Block();
  };
  Block Node::operator+(Block&& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return Block();
  };
  const Node& Node::operator+=(const Node& B) {
    std::cout << "Not implemented!!" << std::endl; abort();
    return *this;
  };
  const Node& Node::operator+=(Block&& B) {
    std::cout << "Not implemented!!" << std::endl; abort();
    return *this;
  };
  Block Node::operator-(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return Block();
  };
  Block Node::operator-(Block&& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return Block();
  };
  const Node& Node::operator-=(const Node& B) {
    std::cout << "Not implemented!!" << std::endl; abort();
    return *this;
  };
  const Node& Node::operator-=(Block&& B) {
    std::cout << "Not implemented!!" << std::endl; abort();
    return *this;
  };
  Block Node::operator*(const Node& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return Block();
  };
  Block Node::operator*(Block&& B) const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return Block();
  };

  const bool Node::is(const int enum_id) const {
    return enum_id == HICMA_NODE;
  }

  const char* Node::is_string() const { return "Node"; }

  double Node::norm() const {
    std::cout << "Not implemented!!" << std::endl; abort();
    return 0.0;
  };

  void Node::print() const {};

  void Node::getrf() {};

  void Node::trsm(const Node& A, const char& uplo) {};

  void Node::gemm(const Node& A, const Node& B) {};

  const block_map_t& Node::get_map(void) const
  {
    return this.block_map;
  };

  void create_dense_block(std::vector<double> &data) {};

  std::vector<Block> get_data(void) {};

  bool has_block(const int i, const int j) const {};

  void single_process_split(const int proc_id) {};

  void multi_process_split(void) {};
}
