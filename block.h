#ifndef block
#define block
#include <memory>

namespace hicma {

class Node;

class Block {
  public:
    std::unique_ptr<Node> ptr;

    Block();

    Block(const Block& ref);
    Block(Block&& ref);
    Block(const Node& ref);
    Block(Node&& ref);

    ~Block();

    // NOTE: Take care to add members new members to swap
    friend void swap(Block& first, Block& second);

    const Block& operator=(Block val);
    const Block& operator=(const Node& ref);
    const Block& operator=(Node&& ref);

    const Block& operator=(double a);

    Block operator+(const Block& rhs) const;
    const Node& operator+=(const Block& rhs);
    Block operator-(const Block& rhs) const;
    const Node& operator-=(const Block& rhs);
    Block operator*(const Block& rhs) const;

    Block operator+(const Node& rhs) const;
    const Node& operator+=(const Node& rhs);
    Block operator-(const Node& rhs) const;
    const Node& operator-=(const Node& rhs);
    Block operator*(const Node& rhs) const;

    const Node& operator[](const int i) const;
    Block& operator[](const int i);

    const Node& operator()(const int i, const int j) const;
    Block& operator()(const int i, const int j);

    const bool is(const int i) const;
    const char* is_string() const;

    void resize(int);

    void resize(int, int);

    double norm() const;

    void print() const;

    void getrf();

    void trsm(const Block& A, const char& uplo);
    void trsm(const Node& A, const char& uplo);

    void gemm(const Block&, const Block&);
    void gemm(const Node&, const Node&);
};
}

#endif
