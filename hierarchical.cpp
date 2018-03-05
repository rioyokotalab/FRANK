#include "dense.h"
#include "low_rank.h"
#include "hierarchical.h"

typedef std::vector<double>::iterator Iter;

namespace hicma {
  Hierarchical::Hierarchical() {
    dim[0]=0; dim[1]=0;
  }

  Hierarchical::Hierarchical(const int m) {
    dim[0]=m; dim[1]=1; data.resize(dim[0]);
  }

  Hierarchical::Hierarchical(const int m, const int n) {
    dim[0]=m; dim[1]=n; data.resize(dim[0]*dim[1]);
  }

  Hierarchical::Hierarchical(
      const size_t max_n_leaf,
      Iter xi_begin,
      Iter xi_end,
      Iter xj_begin,
      Iter xj_end,
      const Hierarchical* parent,
      const int i_rel=0,
      const int j_rel=0
      ) : Node(parent, i_rel, j_rel) {
    dim[0]=2; dim[1]=2;
    data.resize(4);
    size_t xi_half = (xi_end - xi_begin)/2;
    size_t xj_half = (xj_end - xj_begin)/2;

    for ( int i=0; i<2; ++i ) {
      for ( int j=0; j<2; ++j ) {
        int i_tot, j_tot;
        if ( parent ) {
          i_tot = (parent->i << 1) + i;
          j_tot = (parent->j << 1) + j;
        }
        else {
          i_tot = i;
          j_tot = j;
        }
        if ( std::abs(i_tot - j_tot) <= 1 ) {
          if ( xi_half <= max_n_leaf || xj_half <= max_n_leaf ) {
            Dense D(this, i, j, xi_half, xj_half);
            data[i*dim[1] + j] = D;
          }
        }
      }
    }
  }

  boost::any& Hierarchical::operator[](const int i) {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  const boost::any& Hierarchical::operator[](const int i) const {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  boost::any& Hierarchical::operator()(const int i, const int j) {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  const boost::any& Hierarchical::operator()(const int i, const int j) const {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  Dense& Hierarchical::D(const int i) {
    assert(i<dim[0]*dim[1]);
    return boost::any_cast<Dense&>(data[i]);
  }

  Dense& Hierarchical::D(const int i, const int j) {
    assert(i<dim[0] && j<dim[1]);
    return boost::any_cast<Dense&>(data[i*dim[1]+j]);
  }
}
