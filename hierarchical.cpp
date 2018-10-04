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
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
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
            (*this).data[i*dim[1]+j] = Dense(
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
            (*this).data[i*dim[1]+j] = Hierarchical(
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
          (*this).data[i*dim[1]+j] = LowRank(
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

  Hierarchical::Hierarchical(const Hierarchical& A)
    : Node(A.i_abs,A.j_abs,A.level), data(A.data) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1];
  }

  Hierarchical::Hierarchical(Hierarchical&& A) {
    swap(*this, A);
  }

  Hierarchical::Hierarchical(const Hierarchical* A)
    : Node(A->i_abs,A->j_abs,A->level), data(A->data) {
    dim[0]=A->dim[0]; dim[1]=A->dim[1];
    data = A->data;
  }

  Hierarchical::Hierarchical(const Block& A)
    : Node((*A.ptr).i_abs, (*A.ptr).j_abs, (*A.ptr).level) {
    assert(A.is(HICMA_HIERARCHICAL));
    Hierarchical& ref = static_cast<Hierarchical&>(*A.ptr);
    dim[0]=ref.dim[0]; dim[1]=ref.dim[1];
    data = ref.data;
  }

  Hierarchical* Hierarchical::clone() const {
    return new Hierarchical(*this);
  }

  void swap(Hierarchical& A, Hierarchical& B) {
    using std::swap;
    swap(A.data, B.data);
    swap(A.dim, B.dim);
    swap(A.i_abs, B.i_abs);
    swap(A.j_abs, B.j_abs);
    swap(A.level, B.level);
  }

  const Node& Hierarchical::operator=(const Node& _A) {
    if (_A.is(HICMA_HIERARCHICAL)) {
      // This can be avoided if Node has data and dim members!
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      dim[0]=A.dim[0]; dim[1]=A.dim[1];
      data.resize(dim[0]*dim[1]);
      // TODO Explicit constructor is called here! Make sure it's done right,
      // including inheritance
      data = A.data;
      return *this;
    } else {
      std::cerr << this->type() << " = " << _A.type();
      std::cerr << " is undefined." << std::endl;
      return *this;
    }
  }

  const Node& Hierarchical::operator=(Node&& A) {
    if (A.is(HICMA_HIERARCHICAL)) {
      swap(*this, static_cast<Hierarchical&>(A));
      return *this;
    } else {
      std::cerr << this->type() << " = " << A.type();
      std::cerr << " is undefined." << std::endl;
      return *this;
    }
  }

  const Hierarchical& Hierarchical::operator=(Hierarchical A) {
    swap(*this, A);
    return *this;
  }

  const Node& Hierarchical::operator=(Block A) {
    return *this = std::move(*A.ptr);
  }

  const Node& Hierarchical::operator=(const double a) {
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        (*this)(i, j) = a;
      }
    }
    return *this;
  }

  Block Hierarchical::operator+(const Node& _A) const {
    Block A(*this);
    A += _A;
    return A;
  }

  Block Hierarchical::operator+(Block&& A) const {
    return *this + *A.ptr;
  }
  const Node& Hierarchical::operator+=(const Node& _A) {
    if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
      for (int i=0; i<dim[0]; i++)
        for (int j=0; j<dim[1]; j++)
          (*this)(i, j) += A(i, j);
      return *this;
    } else {
      std::cerr << this->type() << " + " << _A.type();
      std::cerr << " is undefined." << std::endl;
      return *this;
    }
  }
  const Node& Hierarchical::operator+=(Block&& A) {
    return *this += *A.ptr;
  }

  Block Hierarchical::operator-(const Node& _A) const {
    Block A(*this);
    A -= _A;
    return A;
  }

  Block Hierarchical::operator-(Block&& A) const {
    return *this - *A.ptr;
  }

  const Node& Hierarchical::operator-=(const Node& _A) {
    if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      assert(dim[0]==A.dim[0] && dim[1]==A.dim[1]);
      for (int i=0; i<dim[0]; i++)
        for (int j=0; j<dim[1]; j++)
          (*this)(i, j) -= A(i, j);
      return *this;
    } else {
      std::cerr << this->type() << " - " << _A.type();
      std::cerr << " is undefined." << std::endl;
      return *this;
    }
  }

  const Node& Hierarchical::operator-=(Block&& A) {
    return *this -= *A.ptr;
  }

  Block Hierarchical::operator*(const Node& _A) const {
    if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      assert(dim[1] == A.dim[0]);
      Hierarchical B(dim[0], A.dim[1]);
      for (int i=0; i<dim[0]; i++) {
        for (int j=0; j<A.dim[1]; j++) {
          B(i, j) = (*this)(i, 0) * A(0, j);
          int rank = -1;
          // Avoid unnecessary recompression by making it dense now
          // and later recompressing it only once
          if (B(i, j).is(HICMA_LOWRANK)) {
            rank = static_cast<LowRank&>(*B(i, j).ptr).rank;
            B(i, j) = static_cast<LowRank&>(*B(i, j).ptr).dense();
          }
          for (int k=1; k<dim[1]; k++) {
            B(i, j) += (*this)(i, k) * A(k, j);
          }
          // If it was LowRank earlier, return it to LowRank now
          if (rank != -1 && A.dim[1] >= rank) {
            B(i, j) = LowRank(B(i, j), rank);
          }
        }
      }
      return B;
    } else {
      std::cerr << this->type() << " * " << _A.type();
      std::cerr << " is undefined." << std::endl;
      return Block();
    }
  }

  Block Hierarchical::operator*(Block&& A) const {
    return *this * *A.ptr;
  }

  const Node& Hierarchical::operator[](const int i) const {
    assert(i<dim[0]*dim[1]);
    return *data[i].ptr;
  }

  Block& Hierarchical::operator[](const int i) {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  const Node& Hierarchical::operator()(const int i, const int j) const {
    assert(i<dim[0] && j<dim[1]);
    return *data[i*dim[1]+j].ptr;
  }

  Block& Hierarchical::operator()(const int i, const int j) {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  const bool Hierarchical::is(const int enum_id) const {
    return enum_id == HICMA_HIERARCHICAL;
  }

  const char* Hierarchical::type() const { return "Hierarchical"; }

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

  void Hierarchical::trsm(const Node& _A, const char& uplo) {
    if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      if (dim[1] == 1) {
        switch (uplo) {
        case 'l' :
          for (int i=0; i<dim[0]; i++) {
            for (int j=0; j<i; j++) {
              (*this)[i].gemm(A(i,j), (*this)[j]);
            }
            (*this)[i].trsm(A(i,i),'l');
          }
          break;
        case 'u' :
          for (int i=dim[0]-1; i>=0; i--) {
            for (int j=dim[0]-1; j>i; j--) {
              (*this)[i].gemm(A(i,j), (*this)[j]);
            }
            (*this)[i].trsm(A(i,i),'u');
          }
          break;
        default :
          std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
          abort();
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
              (*this).gemm_row(A, *this, i, j, 0, i);
              // for (int i_old=0; i_old<i; i_old++) {
              //   (*this)(i,j).gemm(A(i,i_old), (*this)(i_old,j));
              // }
              (*this)(i,j).trsm(A(i,i),'l');
            }
          }
          break;
        case 'u' :
          // Loop over rows, same calculation for all
          for (int i=0; i<dim[0]; i++) {
            // Loop over cols, getting new results
            for (int j=0; j<dim[1]; j++) {
              // Loop over previously calculated col, accumulate results
              (*this).gemm_row(*this, A, i, j, 0, j);
              // for (int j_old=0; j_old<j; j_old++) {
              //   (*this)(i,j).gemm((*this)(i,j_old),A(j_old,j));
              // }
              (*this)(i,j).trsm(A(j,j),'u');
            }
          }
          break;
        default :
          std::cerr << "Second argument must be 'l' for lower, 'u' for upper." << std::endl;
          abort();
        }
      }
    } else {
      std::cerr << this->type() << " /= " << _A.type();
      std::cerr << " is undefined." << std::endl;
      abort();
    }
  }

  void Hierarchical::gemm(const Node& _A, const Node& _B) {
    if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      if (_B.is(HICMA_HIERARCHICAL)) {
        const Hierarchical& B = static_cast<const Hierarchical&>(_B);
        assert(dim[0]==A.dim[0] && dim[1]==B.dim[1]);
        assert(A.dim[1] == B.dim[0]);
        for (int i=0; i<dim[0]; i++) {
          for (int j=0; j<dim[1]; j++) {
            (*this).gemm_row(A, B, i, j, 0, A.dim[1]);
            // for (int k=0; k<A.dim[1]; k++) {
            //   (*this)(i,j).gemm(A(i,k), B(k,j));
            // }
          }
        }
      } else {
        std::cerr << this->type() << " -= " << _A.type();
        std::cerr << " * " << _B.type() << " is undefined." << std::endl;
        abort();
      }
    } else {
      std::cerr << this->type() << " -= " << _A.type();
      std::cerr << " * " << _B.type() << " is undefined." << std::endl;
      abort();
    }
  }

  // TODO: Check if this really avoids recompression
  void Hierarchical::gemm_row(
                              const Hierarchical& A, const Hierarchical& B,
                              const int i, const int j, const int k_min, const int k_max)
  {
    int rank = -1;
    if ((*this)(i, j).is(HICMA_LOWRANK)) {
      rank = static_cast<LowRank&>(*(*this)(i, j).ptr).rank;
      (*this)(i, j) = static_cast<LowRank&>(*(*this)(i, j).ptr).dense();
    }
    for (int k=k_min; k<k_max; k++) {
      (*this)(i, j).gemm(A(i, k), B(k, j));
    }
    // If it was LowRank earlier, return it to LowRank now
    if (rank != -1) {
      (*this)(i, j) = LowRank((*this)(i, j), rank);
    }
  }
}
