#ifndef block_h
#define block_h
#include <cassert>
#include <iostream>
#include <memory>

namespace hicma {

  class Node;

  class Block {
  public:
    // NOTE: Take care to add members new members to swap
    std::unique_ptr<Node> ptr;

    Block();

    Block(const Block& A);

    Block(Block&& A);

    Block(const Node& A);

    Block(Node&& A);

    ~Block();

    friend void swap(Block& A, Block& B);

    const Block& operator=(Block A);

    const Block& operator=(const Node& A);

    const Block& operator=(Node&& A);

    const Block& operator=(double a);

    const Node& operator[](const int i) const;

    Block& operator[](const int i);

    const Node& operator()(const int i, const int j) const;

    Block& operator()(const int i, const int j);

    const bool is(const int i) const;

    const char* type() const;

    double norm() const;

    void print() const;

    void getrf();

    void trsm(const Block& A, const char& uplo);

    void trsm(const Node& A, const char& uplo);

    void gemm(const Block& A, const Block& B);

    void gemm(const Node& A, const Node& B);
  };
}

#endif
