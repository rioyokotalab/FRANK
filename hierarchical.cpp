#include <algorithm>
#include "hblas.h"

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
                             void (*func)(
                                          std::vector<double>& data,
                                          std::vector<double>& x,
                                          const int& ni,
                                          const int& nj,
                                          const int& i_begin,
                                          const int& j_begin
                                          ),
                             std::vector<double>& x,
                             const int ni,
                             const int nj,
                             const int rank,
                             const int nleaf,
                             const int admis=1,
                             const int ni_level=0,
                             const int nj_level=0,
                             const int i_begin=0,
                             const int j_begin=0,
                             const int i_abs=0,
                             const int j_abs=0,
                             const int level=0
                             ) {
    if ( !level ) {
      assert(int(x.size()) == std::max(ni,nj));
      std::sort(x.begin(),x.end());
    }
    dim[0] = std::min(ni_level,ni);
    dim[1] = std::min(nj_level,nj);
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
        if ( std::abs(i_abs_child - j_abs_child) <= admis ) { // TODO: use x in admissibility condition
          if ( ni_child <= nleaf && nj_child <= nleaf ) {
            Dense D(func, x, ni_child, nj_child, i_begin_child, j_begin_child);
            (*this)(i,j) = D;
          }
          else {
            Hierarchical H(
                           func,
                           x,
                           ni_child,
                           nj_child,
                           rank,
                           nleaf,
                           admis,
                           ni_level,
                           nj_level,
                           i_begin_child,
                           j_begin_child,
                           i_abs_child,
                           j_abs_child,
                           level+1
                           );
            (*this)(i,j) = H;
          }
        }
        else {
          Dense D(func, x, ni_child, nj_child, i_begin_child, j_begin_child);
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

  const Hierarchical& Hierarchical::operator=(const Hierarchical& A) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1];
    data.resize(dim[0]*dim[1]);
    data = A.data;
    return *this;
  }

  const Hierarchical Hierarchical::operator+=(const Hierarchical& A) {
    assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        add((*this)(i,j), A(i,j), (*this)(i,j));
      }
    }
    return *this;
  }

  const Hierarchical Hierarchical::operator-=(const Hierarchical& A) {
    assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
    for (int i=0; i<dim[0]; i++)
      for (int j=0; j<dim[1]; j++)
        sub((*this)(i,j), A(i,j), (*this)(i,j));
    return *this;
  }

  const Hierarchical Hierarchical::operator*=(const Hierarchical& A) {
    assert(dim[1] == A.dim[0]);
    Hierarchical B(dim[0],A.dim[1]);
    for (int i=0; i<dim[0]; i++)
      for (int j=0; j<A.dim[1]; j++)
        for (int k=0; k<dim[1]; k++)
          gemm((*this)(i,k), A(k,j), B(i,j));
    return B;
  }

  Hierarchical Hierarchical::operator+(const Hierarchical& A) const {
    return Hierarchical(*this) += A;
  }

  Hierarchical Hierarchical::operator-(const Hierarchical& A) const {
    return Hierarchical(*this) -= A;
  }

  Hierarchical Hierarchical::operator*(const Hierarchical& A) const {
    return Hierarchical(*this) *= A;
  }

  Dense& Hierarchical::dense(const int i) {
    assert(i<dim[0]*dim[1]);
    return boost::any_cast<Dense&>(data[i]);
  }

  Dense& Hierarchical::dense(const int i, const int j) {
    assert(i<dim[0] && j<dim[1]);
    return boost::any_cast<Dense&>(data[i*dim[1]+j]);
  }

  std::vector<int> Hierarchical::getrf() {
    std::vector<int>(a);
    return a;
  }
}
