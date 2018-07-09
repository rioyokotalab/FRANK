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

    const bool is(const int) const;

    const char* is_string() const;

    double norm() const;

    void print() const;

    void getrf();

    void trsm(const BlockPtr<T>&, const char&);

    void gemm(const BlockPtr<T>&, const BlockPtr<T>&);
  };

}

#endif
