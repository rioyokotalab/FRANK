#ifndef hierarchical_h
#define hierarchical_h
#include <boost/any.hpp>
#include "node.h"

typedef std::vector<double>::iterator Iter;

namespace hicma {
  class Node;
  class Dense;
  class LowRank;
  class Hierarchical : public Node {
  public:
    int dim[2];
    std::vector<boost::any> data;

    Hierarchical();

    Hierarchical(const int m);

    Hierarchical(const int m, const int n);

    Hierarchical(
                 std::vector<double>& x,
                 const int ni,
                 const int nj,
                 const int rank,
                 const int nleaf,
                 const int i_begin,
                 const int j_begin,
                 const Hierarchical* parent,
                 const int i_rel,
                 const int j_rel
                 );

    boost::any& operator[](const int i);

    const boost::any& operator[](const int i) const;

    boost::any& operator()(const int i, const int j);

    const boost::any& operator()(const int i, const int j) const;

    Dense& D(const int i);

    Dense& D(const int i, const int j);
  };
}
#endif
