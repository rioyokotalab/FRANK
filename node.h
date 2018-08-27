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

  class _Node;
  typedef BlockPtr<_Node> Node;
  class _Hierarchical;
  typedef BlockPtr<_Hierarchical> Hierarchical;
  class _Dense;
  typedef BlockPtr<_Dense> Dense;
  class _LowRank;
  typedef BlockPtr<_LowRank> LowRank;

  class _Node {
  public:
    const int i_abs;
    const int j_abs;
    const int level;
    _Node() : i_abs(0), j_abs(0), level(0) {}
    _Node(
        const int _i_abs,
        const int _j_abs,
        const int _level) : i_abs(_i_abs), j_abs(_j_abs), level(_level) {}

    // TODO Check if this is really necessary (became necessary when adding
    // explicit constructor for BlockPtr using Args forwarding.
    // Called when copying data = A.data in Hierarchical::mul
    _Node(Node) : i_abs(0), j_abs(0), level(0) {}

    virtual ~_Node();

    virtual _Node* clone() const;

    // TODO Change this once other = is not needed anymore
    virtual const _Node& operator=(const double a);

    // TODO remove
    virtual const _Node& operator=(const _Node& A);

    virtual const _Node& operator=(const Node& A);

    virtual const bool is(const int enum_id) const;

    virtual const char* is_string() const;

    virtual Node add(const Node& B) const;

    virtual Node sub(const Node& B) const;

    virtual Node mul(const Node& B) const;

    virtual double norm() const;

    virtual void print() const;

    virtual void getrf();

    virtual void trsm(const Node& A, const char& uplo);

    virtual void gemm(const Node& A, const Node& B);
  };

  Node operator+(const Node& A, const Node& B);

  // This version seems const correct, but
  // const Node& operator+=(const Node& A, const Node& B)
  // also works and might be preferable (speed?)
  const Node operator+=(const Node& A, const Node& B);

  Node operator-(const Node& A, const Node& B);

  // This version seems const correct, but
  // const Node& operator-=(const Node& A, const Node& B)
  // also works and might be preferable (speed?)
  Node operator-=(const Node& A, const Node& B);

  Node operator*(const Node& A, const Node& B);

}
#endif
