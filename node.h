#ifndef node_h
#define node_h
#include <vector>

namespace hicma {
  class Node {
  public:
    int i;
    int j;
    int level;
    //virtual double& operator[](int) {double dummy; return dummy;}
    //virtual Node* operator[](int) {Node* dummy; return dummy;}
  };
}
#endif
