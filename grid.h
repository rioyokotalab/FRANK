#ifndef grid_h
#define grid_h
#include "dense.h"
#include "low_rank.h"

namespace hicma {
  class Grid {
  public:
    struct Data {
      int flag;
      Dense d;
      LowRank l;
    }
    int dim[2];
    std::vector<Data> data;

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

    Data operator[](const int i) {
      return data[i];
    }
    Data operator()(const int i, const int j) {
      return data[i*dim[1]+j];
    }
  };
}
#endif
