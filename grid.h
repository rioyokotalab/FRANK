#ifndef grid_h
#define grid_h
#include "node.h"

namespace hicma {
  class Grid : public Node {
  public:
    int dim[2];
    std::vector<Node*> data;

    Grid(int m) {
      dim[0] = m;
      dim[1] = 1;
      data.resize(m);
    }

    Grid(int m, int n) {
      dim[0] = m;
      dim[1] = n;
      data.resize(m*n);
    }

    Node* operator[](const int i) {
      return data[i];
    }
    Node* operator()(const int i, const int j) {
      return data[i*dim[1]+j];
    }
  };
}
#endif
