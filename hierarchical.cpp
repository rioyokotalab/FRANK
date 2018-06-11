#include <algorithm>
#include <memory>
#include <cassert>
#include "node.h"
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

  Hierarchical::Hierarchical(const Hierarchical& A) : Node(A.i_abs,A.j_abs,A.level), data(A.data) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1];
    data.resize(dim[0]*dim[1]);
    for ( int i=0; i<dim[0]; i++ ) {
      for ( int j=0; j<dim[1]; j++ ) {
        data[i*dim[1] + j] = std::shared_ptr<Node>(
            (*A.data[i*dim[1] + j]).clone());
      }
    }
  }

  Hierarchical::Hierarchical(const Hierarchical* A) : Node(A->i_abs,A->j_abs,A->level), data(A->data) {
    dim[0]=A->dim[0]; dim[1]=A->dim[1];
    data.resize(dim[0]*dim[1]);
    for ( int i=0; i<dim[0]; i++ ) {
      for ( int j=0; j<dim[1]; j++ ) {
        data[i*dim[1] + j] = std::shared_ptr<Node>(
            (*A->data[i*dim[1] + j]).clone());
      }
    }
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
                             const int admis,
                             const int ni_level,
                             const int nj_level,
                             const int i_begin,
                             const int j_begin,
                             const int _i_abs,
                             const int _j_abs,
                             const int _level
                             ) : Node(_i_abs,_j_abs,_level) {
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
        if (
            // Check regular admissibility
            std::abs(i_abs_child - j_abs_child) <= admis
            // Check if vector, and if so do not use LowRank
            || (nj == 1 || ni == 1) /* Check if vector */ ) { // TODO: use x in admissibility condition
          if ( ni_child <= nleaf && nj_child <= nleaf ) {
            (*this).data[i*dim[1]+j] = std::make_shared<Dense>(
                func,
                x,
                ni_child,
                nj_child,
                i_begin_child,
                j_begin_child,
                i_abs_child,
                j_abs_child,
                level+1);
          }
          else {
            (*this).data[i*dim[1]+j] = std::make_shared<Hierarchical>(
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
                level+1);
          }
        }
        else {
          (*this).data[i*dim[1]+j] = std::make_shared<LowRank>(
            Dense(
                  func,
                  x,
                  ni_child,
                  nj_child,
                  i_begin_child,
                  j_begin_child,
                  i_abs_child,
                  j_abs_child,
                  level+1)
            , rank);// TODO : create a LowRank constructor that does ID with x
        }
      }
    }
  }

  Hierarchical* Hierarchical::clone() const {
    return new Hierarchical(*this);
  }

  const bool Hierarchical::is(const int enum_id) const {
    return enum_id == HICMA_HIERARCHICAL;
  }

  const char* Hierarchical::is_string() const { return "Hierarchical"; }

  Node& Hierarchical::operator[](const int i) {
    assert(i<dim[0]*dim[1]);
    return *data[i];
  }

  const Node& Hierarchical::operator[](const int i) const {
    assert(i<dim[0]*dim[1]);
    return *data[i];
  }

  Node& Hierarchical::operator()(const int i, const int j) {
    assert(i<dim[0] && j<dim[1]);
    return *data[i*dim[1]+j];
  }

  const Node& Hierarchical::operator()(const int i, const int j) const {
    assert(i<dim[0] && j<dim[1]);
    return *data[i*dim[1]+j];
  }

  const Node& Hierarchical::operator=(const double a) {
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        (*this)(i,j) = a;
      }
    }
    return *this;
  }

  const Node& Hierarchical::operator=(const Node& A) {
    if (A.is(HICMA_HIERARCHICAL)) {
      // This can be avoided if Node has data and dim members!
      const Hierarchical& AR = static_cast<const Hierarchical&>(A);
      dim[0]=AR.dim[0]; dim[1]=AR.dim[1];
      data.resize(dim[0]*dim[1]);
      data = AR.data;
      return *this;
    } else {
      std::cout << this->is_string() << " = " << A.is_string();
      std::cout << " not implemented!" << std::endl;
      return *this;
    }
  }

  const Node& Hierarchical::operator=(const std::shared_ptr<Node> A) {
    return *this = *A;
  }

  std::shared_ptr<Node> Hierarchical::add(const Node& B) const {
    if (B.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& BR = static_cast<const Hierarchical&>(B);
      assert(dim[0]==BR.dim[0] && dim[1]==BR.dim[1]);
      std::shared_ptr<Hierarchical> Out = std::make_shared<Hierarchical>(*this);
      for (int i=0; i<dim[0]; i++)
        for (int j=0; j<dim[1]; j++)
          (*Out)(i,j) += BR(i,j);
      return Out;
    } else {
        std::cout << this->is_string() << " + " << B.is_string();
        std::cout << " is undefined!" << std::endl;
        return std::shared_ptr<Node>(nullptr);
    }
  }

  std::shared_ptr<Node> Hierarchical::sub(const Node& B) const {
    if (B.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& BR = static_cast<const Hierarchical&>(B);
      assert(dim[0]==BR.dim[0] && dim[1]==BR.dim[1]);
      std::shared_ptr<Hierarchical> Out = std::make_shared<Hierarchical>(*this);
      for (int i=0; i<dim[0]; i++)
        for (int j=0; j<dim[1]; j++)
          (*Out)(i,j) -= BR(i,j);
      return Out;
    } else {
        std::cout << this->is_string() << " - " << B.is_string();
        std::cout << " is undefined!" << std::endl;
        return std::shared_ptr<Node>(nullptr);
    }
  }

  std::shared_ptr<Node> Hierarchical::mul(const Node& B) const {
    if (B.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& BR = static_cast<const Hierarchical&>(B);
      assert(dim[1] == BR.dim[0]);
      std::shared_ptr<Hierarchical> Out = std::make_shared<Hierarchical>(BR);
      //std::cout << BR.norm_test() << std::endl;
      (*Out) = 0;
      //std::cout << BR.norm_test() << std::endl;
      for (int i=0; i<dim[0]; i++) {
        for (int j=0; j<BR.dim[1]; j++) {
          for (int k=0; k<dim[1]; k++) {
            //std::cout << (*((*this)(i,k) * BR(k,j))).norm_test() << std::endl;
            (*Out)(i,j) += (*this)(i,k) * BR(k,j);
          }
        }
      }
      return Out;
    } else {
        std::cout << this->is_string() << " * " << B.is_string();
        std::cout << " is undefined!" << std::endl;
        return std::shared_ptr<Node>(nullptr);
    }
  }

  double Hierarchical::norm() const {
    double l2 = 0;
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        l2 += (*this)(i,j).norm();
      }
    }
    return l2;
  }

  void Hierarchical::print() const {
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        if ((*this)(i,j).is(HICMA_DENSE)) {
          std::cout << "D (" << i << "," << j << ")" << std::endl;
        }
        else if ((*this)(i,j).is(HICMA_LOWRANK)) {
          std::cout << "L (" << i << "," << j << ")" << std::endl;
        }
        else if ((*this)(i,j).is(HICMA_HIERARCHICAL)) {
          std::cout << "H (" << i << "," << j << ")" << std::endl;
        }
        else {
          std::cout << "? (" << i << "," << j << ")" << std::endl;
        }
        (*this)(i,j).print();
      }
      std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
  }

  void Hierarchical::getrf() {
    for (int i=0; i<dim[0]; i++) {
      (*this)(i,i).getrf();
      for (int j=i+1; j<dim[0]; j++) {
        (*this)(i,j).trsm((*this)(i,i),'l');
        (*this)(j,i).trsm((*this)(i,i),'u');
      }
      for (int j=i+1; j<dim[0]; j++) {
        for (int k=i+1; k<dim[0]; k++) {
          (*this)(j,k).gemm((*this)(j,i),(*this)(i,k));
        }
      }
    }
  }

  void Hierarchical::trsm(const Node& A, const char& uplo) {
    if (A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& AR = static_cast<const Hierarchical&>(A);
      if (dim[1] == 1) {
        switch (uplo) {
        case 'l' :
          for (int i=0; i<dim[0]; i++) {
            for (int j=0; j<i; j++) {
              (*this)[i].gemm(AR(i,j), (*this)[j]);
            }
            (*this)[i].trsm(AR(i,i),'l');
          }
          break;
        case 'u' :
          for (int i=dim[0]-1; i>=0; i--) {
            for (int j=dim[0]-1; j>i; j--) {
              (*this)[i].gemm(AR(i,j), (*this)[j]);
            }
            (*this)[i].trsm(AR(i,i),'u');
          }
          break;
        default :
          fprintf(stderr,"Second argument must be 'l' for lower, 'u' for upper.\n"); abort();
        }
      }
      else {
        switch (uplo) {
        case 'l' :
          // Loop over cols, same calculation for all
          for (int j=0; j<dim[1]; j++) {
            // Loop over rows, getting new results
            for (int i=0; i<dim[0]; i++) {
              // Loop over previously calculated row, accumulate results
              for (int i_old=0; i_old<i; i_old++) {
                (*this)(i,j).gemm(AR(i,i_old), (*this)(i_old,j));
              }
              (*this)(i,j).trsm(AR(i,i),'l');
            }
          }
          break;
        case 'u' :
          // Loop over rows, same calculation for all
          for (int i=0; i<dim[0]; i++) {
            // Loop over cols, getting new results
            for (int j=0; j<dim[1]; j++) {
              // Loop over previously calculated col, accumulate results
              for (int j_old=0; j_old<j; j_old++) {
                (*this)(i,j).gemm((*this)(i,j_old),AR(j_old,j));
              }
              (*this)(i,j).trsm(AR(j,j),'u');
            }
          }
          break;
        default :
          fprintf(stderr,"Second argument must be 'l' for lower, 'u' for upper.\n"); abort();
        }
      }
    } else {
      fprintf(
          stderr,"%s /= %s undefined.\n",
          this->is_string(), A.is_string());
      abort();
    }
  }

  void Hierarchical::gemm(const Node& A, const Node& B) {
    if (A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& AR = static_cast<const Hierarchical&>(A);
      if (B.is(HICMA_HIERARCHICAL)) {
        const Hierarchical& BR = static_cast<const Hierarchical&>(B);
        assert(dim[0]==AR.dim[0] && dim[1]==BR.dim[1]);
        assert(AR.dim[1] == BR.dim[0]);
        for (int i=0; i<dim[0]; i++) {
          for (int j=0; j<dim[1]; j++) {
            for (int k=0; k<AR.dim[1]; k++) {
              (*this)(i,j).gemm(AR(i,k), BR(k,j));
            }
          }
        }
      } else {
        fprintf(
            stderr,"%s += %s * %s undefined.\n",
            this->is_string(), A.is_string(), B.is_string());
        abort();
      }
    } else {
      fprintf(
          stderr,"%s += %s * %s undefined.\n",
          this->is_string(), A.is_string(), B.is_string());
      abort();
    }
  }
}
