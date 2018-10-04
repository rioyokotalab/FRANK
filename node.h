#ifndef node_h
#define node_h
#include "block.h"

#include <vector>

namespace hicma {

  // Used in polymorphic code to know what actually a class is.
  enum {
    HICMA_NODE,
    HICMA_HIERARCHICAL,
    HICMA_LOWRANK,
    HICMA_DENSE
  };

  class Node {
  public:
    /// Row number of the node on the current recursion level
    int i_abs;
    /// Column number of the node on the current recursion level
    int j_abs;
    /// Recursion level of the node
    int level;

    Node();

    Node(const int _i_abs, const int _j_abs, const int _level);

    Node(const Node& ref);

    virtual ~Node();

    virtual Node* clone() const;

    // NOTE: Take care to add members new members to swap
    friend void swap(Node& first, Node& second);

    virtual const Node& operator=(const double a);

    virtual const Node& operator=(const Node& A);

    virtual const Node& operator=(Node&& A);

    virtual const Node& operator=(Block A);

    virtual Block operator+(const Node& A) const;

    virtual Block operator+(Block&& A) const;

    virtual const Node& operator+=(const Node& A);

    virtual const Node& operator+=(Block&& A);

    virtual Block operator-(const Node& A) const;

    virtual Block operator-(Block&& A) const;

    virtual const Node& operator-=(const Node& A);

    virtual const Node& operator-=(Block&& A);

    virtual Block operator*(const Node& A) const;

    virtual Block operator*(Block&& A) const;

    virtual const bool is(const int enum_id) const;

    virtual const char* is_string() const;

    virtual double norm() const;

    virtual void print() const;

    virtual void getrf();

    virtual void trsm(const Node& A, const char& uplo);

    virtual void gemm(const Node& A, const Node& B);
  };
}
#endif
