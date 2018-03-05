#ifndef node_h
#define node_h
#include <boost/any.hpp>

namespace hicma {
  class Hierarchical;
  class Node {
  protected:
    int i;
    int j;
    int level;

    // Root constructor
    Node();
    // Non-root constructor
    Node(const Hierarchical*, const int, const int);
  };
}
#endif
