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

  class Node;
  typedef BlockPtr<Node> NodePtr;
  class Hierarchical;
  typedef BlockPtr<Hierarchical> HierarchicalPtr;
  class Dense;
  typedef BlockPtr<Dense> DensePtr;
  class LowRank;
  typedef BlockPtr<LowRank> LowRankPtr;

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

    // TODO Check if this is really necessary (became necessary when adding
    // explicit constructor for BlockPtr using Args forwarding.
    // Called when copying data = A.data in Hierarchical::mul
    Node(NodePtr) : i_abs(0), j_abs(0), level(0) {}

    virtual ~Node();

    virtual Node* clone() const;

    // TODO Change this once other = is not needed anymore
    virtual const Node& operator=(const double a);

    // TODO remove
    virtual const Node& operator=(const Node& A);

    virtual const Node& operator=(const NodePtr& A);

    virtual const bool is(const int enum_id) const;

    virtual const char* is_string() const;

    virtual NodePtr add(const NodePtr& B) const;

    virtual NodePtr sub(const NodePtr& B) const;

    virtual NodePtr mul(const NodePtr& B) const;

    virtual double norm() const;

    virtual void print() const;

    virtual void getrf();

    virtual void trsm(const NodePtr& A, const char& uplo);

    virtual void gemm(const NodePtr& A, const NodePtr& B);
  };

  NodePtr operator+(const NodePtr& A, const NodePtr& B);

  // This version seems const correct, but
  // const NodePtr& operator+=(const NodePtr& A, const NodePtr& B)
  // also works and might be preferable (speed?)
  const NodePtr operator+=(const NodePtr A, const NodePtr& B);

  NodePtr operator-(const NodePtr& A, const NodePtr& B);

  // This version seems const correct, but
  // const NodePtr& operator-=(const NodePtr& A, const NodePtr& B)
  // also works and might be preferable (speed?)
  NodePtr operator-=(NodePtr A, const NodePtr& B);

  NodePtr operator*(const NodePtr& A, const NodePtr& B);

}
#endif
