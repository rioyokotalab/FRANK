#ifndef node_h
#define node_h
#include <boost/any.hpp>
#include <vector>

namespace hicma {
  class Hierarchical;
  class Node {
  public:
    int i_abs;
    int j_abs;
    int level;

    Node();

    Node(const Hierarchical*, const int, const int);
  };
}
#endif
