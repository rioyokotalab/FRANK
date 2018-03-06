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
                             const Hierarchical* parent=NULL,
                             const int i_rel=0,
                             const int j_rel=0
                             ) : Node(parent, i_rel, j_rel) {
    if ( parent ) {
      assert(int(x.size()) == nj);
      std::sort(x.begin(),x.end());
    }
    dim[0]=2; dim[1]=2;
    data.resize(4);
    for ( int i_rel_child=0; i_rel_child<2; i_rel_child++ ) {
      for ( int j_rel_child=0; j_rel_child<2; j_rel_child++ ) {
        int ni_child, nj_child;
        if ( i_rel_child == 0 ) ni_child = ni/2;
        else ni_child = ni - ni/2;
        if ( j_rel_child == 0 ) nj_child = nj/2;
        else nj_child = nj - nj/2;
        int i_begin_child = i_begin + ni/2 * i_rel_child;
        int j_begin_child = j_begin + nj/2 * j_rel_child;
        int i_abs_child, j_abs_child;
        if ( parent ) {
          i_abs_child = (parent->i_abs << 1) + i_rel_child;
          j_abs_child = (parent->j_abs << 1) + j_rel_child;
        }
        else {
          i_abs_child = i_rel_child;
          j_abs_child = j_rel_child;
        }
        if ( std::abs(i_abs_child - j_abs_child) <= 1 ) { // TODO: use x in admissibility condition
          if ( ni <= nleaf && nj <= nleaf ) {
            Dense D(x, ni_child, nj_child, i_begin_child, j_begin_child, this, i_rel_child, j_rel_child);
            (*this)(i_rel_child,j_rel_child) = D;
          }
          else {
            Hierarchical H(x, ni_child, nj_child, rank, nleaf, i_begin_child, j_begin_child, this, i_rel_child, j_rel_child);
            (*this)(i_rel_child,j_rel_child) = H;
          }
        }
        else {
          Dense D(x, ni_child, nj_child, i_begin_child, j_begin_child, this, i_rel_child, j_rel_child);
          LowRank LR(D, rank); // TODO : create a LowRank constructor that does ID with x
          (*this)(i_rel_child,j_rel_child) = LR;
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
