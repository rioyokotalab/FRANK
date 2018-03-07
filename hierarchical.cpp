#include "dense.h"
#include "low_rank.h"
#include "hierarchical.h"

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
                             std::vector<double>& x,
                             const int ni,
                             const int nj,
                             const int rank,
                             const int nleaf,
                             const int i_begin=0,
                             const int j_begin=0,
                             const int i_abs=0,
                             const int j_abs=0,
                             const int level=0
                             ) {
    if ( !level ) {
      assert(int(x.size()) == nj);
      std::sort(x.begin(),x.end());
    }
    dim[0]=2; dim[1]=2;
    data.resize(dim[0]*dim[1]);
    for ( int i=0; i<dim[0]; i++ ) {
      for ( int j=0; j<dim[1]; j++ ) {
        int ni_child = ni/dim[0];
        if ( i == dim[0]-1 ) ni_child = ni - (ni/dim[0]) * (dim[0]-1);
        int nj_child = nj/dim[1];
        if ( j == dim[1]-1 ) nj_child = nj - (nj/dim[1]) * (dim[1]-1);
        int i_begin_child = i_begin + ni/dim[0] * i;
        int j_begin_child = j_begin + nj/dim[1] * j;
        int i_abs_child = i_abs * dim[0] + i;
        int j_abs_child = j_abs * dim[1] + j;
        if ( std::abs(i_abs_child - j_abs_child) <= 1 ) { // TODO: use x in admissibility condition
          if ( ni <= nleaf && nj <= nleaf ) {
            Dense D(x, ni_child, nj_child, i_begin_child, j_begin_child);
            (*this)(i,j) = D;
          }
          else {
            Hierarchical H(x, ni_child, nj_child, rank, nleaf, i_begin_child, j_begin_child, i_abs_child, j_abs_child, level+1);
            (*this)(i,j) = H;
          }
        }
        else {
          Dense D(x, ni_child, nj_child, i_begin_child, j_begin_child);
          LowRank LR(D, rank); // TODO : create a LowRank constructor that does ID with x
          (*this)(i,j) = LR;
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
