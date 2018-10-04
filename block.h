#ifndef block
#define block
#include <memory>

namespace hicma {

  class Node;

  class Block {
  public:
    std::unique_ptr<Node> ptr;

    Block();

    Block(const Block& A);

    Block(Block&& A);

    Block(const Node& A);

    Block(Node&& A);

    ~Block();

    // NOTE: Take care to add members new members to swap
    friend void swap(Block& first, Block& second);

    const Block& operator=(Block A);

    const Block& operator=(const Node& A);

    const Block& operator=(Node&& A);

    const Block& operator=(double a);

    Block operator+(const Block& A) const;

    const Node& operator+=(const Block& A);

    Block operator-(const Block& A) const;

    const Node& operator-=(const Block& A);

    Block operator*(const Block& A) const;

    Block operator+(const Node& A) const;

    const Node& operator+=(const Node& A);

    Block operator-(const Node& A) const;

    const Node& operator-=(const Node& A);

    Block operator*(const Node& A) const;

    const Node& operator[](const int i) const;

    Block& operator[](const int i);

    const Node& operator()(const int i, const int j) const;

    Block& operator()(const int i, const int j);

    const bool is(const int i) const;

    const char* is_string() const;

    void resize(int i);

    void resize(int i, int j);

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
