#ifndef node_h
#define node_h

#include "yorel/multi_methods.hpp"

namespace hicma {

  class Dense;
  class LowRank;
  class Hierarchical;

  class Node : public yorel::multi_methods::selector {
  public:
    MM_CLASS(Node);
    int i_abs; //! Row number of the node on the current recursion level
    int j_abs; //! Column number of the node on the current recursion level
    int level; //! Recursion level of the node

    Node();

    Node(const int _i_abs, const int _j_abs, const int _level);

    Node(const Node& A);

    virtual ~Node();

    virtual Node* clone() const;

    const Node& operator=(Node&& A);

    virtual const char* type() const;

    virtual double norm() const;

    virtual void print() const;

    virtual void transpose();

    // virtual void geqrt(Dense& T);

    // virtual void geqrt(Hierarchical& T);

  };

}
#endif
