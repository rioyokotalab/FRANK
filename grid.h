#ifndef grid_h
#define grid_h
#include <boost/any.hpp>
#include "dense.h"
#include "low_rank.h"

namespace hicma {
  class Grid {
  public:
    int dim[2];
    std::vector<boost::any> data;

    Grid() {
      dim[0]=0; dim[1]=0;
    }
    
    Grid(const int m) {
      dim[0]=m; dim[1]=1; data.resize(dim[0]);
    }

    Grid(const int m, const int n) {
      dim[0]=m; dim[1]=n; data.resize(dim[0]*dim[1]);
    }

    boost::any& operator[](const int i) {
      assert(i<dim[0]*dim[1]);
      return data[i];
    }
    
    const boost::any& operator[](const int i) const {
      assert(i<dim[0]*dim[1]);
      return data[i];
    }
    
    boost::any& operator()(const int i, const int j) {
      assert(i<dim[0] && j<dim[1]);
      return data[i*dim[1]+j];
    }

    const boost::any& operator()(const int i, const int j) const {
      assert(i<dim[0] && j<dim[1]);
      return data[i*dim[1]+j];
    }

    Dense& D(const int i) {
      assert(i<dim[0]*dim[1]);
      return boost::any_cast<Dense&>(data[i]);
    }

    Dense& D(const int i, const int j) {
      assert(i<dim[0] && j<dim[1]);
      return boost::any_cast<Dense&>(data[i*dim[1]+j]);
    }
  };
}
#endif
