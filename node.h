#ifndef node_h
#define node_h
#include <iostream>
#include "block.h"

namespace hicma {

  enum {
    HICMA_NODE,
    HICMA_HIERARCHICAL,
    HICMA_LOWRANK,
    HICMA_DENSE
  };

  class Node {
  public:
    int i_abs;
    int j_abs;
    int level;
    Node();
    Node(const int _i_abs, const int _j_abs, const int _level);

    Node(const Node& ref);

    virtual ~Node();

    virtual Node* clone() const;

    friend void swap(Node& first, Node& second);

    virtual const Node& operator=(const double a);

    virtual const Node& operator=(const Node& A);
    virtual const Node& operator=(Node&& A);

    virtual const Node& operator=(Block A);

    virtual Block operator+(const Node& B) const;
    virtual Block operator+(Block&& B) const;
    virtual const Node& operator+=(const Node& B);
    virtual const Node& operator+=(Block&& B);
    virtual Block operator-(const Node& B) const;
    virtual Block operator-(Block&& B) const;
    virtual const Node& operator-=(const Node& B);
    virtual const Node& operator-=(Block&& B);
    virtual Block operator*(const Node& B) const;
    virtual Block operator*(Block&& B) const;

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
