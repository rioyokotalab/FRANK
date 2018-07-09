#ifndef block_ptr
#define block_ptr
#include <iostream>
#include <memory>

namespace hicma {

  class Node;

  template <typename T = Node>
  class BlockPtr : public std::shared_ptr<T> {
  public:
    template <typename U>
    friend class BlockPtr;

    BlockPtr();

    BlockPtr(nullptr_t);

    BlockPtr(std::shared_ptr<T>);

    BlockPtr(T*);

    template <typename U>
    BlockPtr(std::shared_ptr<U> ptr) : std::shared_ptr<T>(ptr) {};

    double norm();

    void print();

    void getrf();

    void trsm(const Node&, const char&);

    void gemm(const Node&, const Node&);

    void gemm(const Node&, BlockPtr<T>);
  };

}

#endif
