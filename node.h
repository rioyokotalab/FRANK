#ifndef node_h
#define node_h
#include <iostream>
#include "block_ptr.h"

namespace hicma {

  enum {
    HICMA_NODE,
    HICMA_HIERARCHICAL,
    HICMA_LOWRANK,
    HICMA_DENSE
  };

  typedef BlockPtr<Node> NodePtr;

  class Node {
  public:
    const int i_abs;
    const int j_abs;
    const int level;
    Node() : i_abs(0), j_abs(0), level(0) {}
    Node(
        const int _i_abs,
        const int _j_abs,
        const int _level) : i_abs(_i_abs), j_abs(_j_abs), level(_level) {}

    virtual ~Node();

    virtual Node* clone() const;

    // TODO Change this once other = is not needed anymore
    virtual const Node& operator=(const double a);

    virtual const Node& operator=(const Node& A);

    virtual const Node& operator=(const NodePtr A);

    virtual const bool is(const int enum_id) const;

    virtual const char* is_string() const;

    virtual NodePtr add(const Node& B) const;

    virtual NodePtr sub(const Node& B) const;

    virtual NodePtr mul(const Node& B) const;

    virtual double norm() const;

    virtual void print() const;

    virtual void getrf();

    virtual void trsm(const Node& A, const char& uplo);

    virtual void gemm(const Node& A, const Node& B);
  };

  NodePtr operator+(const Node& A, const Node& B);

  NodePtr operator+(const Node& A, const NodePtr B);

  NodePtr operator+(const NodePtr A, const Node& B);

  NodePtr operator+(
      const NodePtr A,
      const NodePtr B);

  const Node& operator+=(Node& A, const NodePtr B);

  const NodePtr operator+=(NodePtr A, const NodePtr B);

  const Node& operator+=(Node& A, const Node& B);

  NodePtr operator-(const Node& A, const Node& B);

  NodePtr operator-(const Node& A, const NodePtr B);

  NodePtr operator-(const NodePtr A, const Node& B);

  NodePtr operator-(
      const NodePtr A,
      const NodePtr B);

  const Node& operator-=(Node& A, const NodePtr B);

  const Node& operator-=(Node& A, const Node& B);

  NodePtr operator*(const Node& A, const Node& B);

  NodePtr operator*(const Node& A, const NodePtr B);

  NodePtr operator*(const NodePtr A, const Node& B);

  NodePtr operator*(
      const NodePtr A,
      const NodePtr B);

  const Node& operator*=(Node& A, const NodePtr B);

  const Node& operator*=(Node& A, const Node& B);

}
#endif
