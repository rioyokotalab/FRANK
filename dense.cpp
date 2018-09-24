#include <iomanip>
//#include <lapacke.h>
#include "dense.h"
#include "low_rank.h"
#include "block.h"

namespace hicma {
  Dense::Dense() {
    dim[0]=0; dim[1]=0;
  }

  Dense::Dense(const int m) {
    dim[0]=m; dim[1]=1; data.resize(dim[0]);
  }

  Dense::Dense(const int m, const int n) {
    dim[0]=m; dim[1]=n; data.resize(dim[0]*dim[1]);
  }

  Dense::Dense(
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
               const int i_begin,
               const int j_begin,
               const int _i_abs,
               const int _j_abs,
               const int _level
               ) : Node(_i_abs,_j_abs,_level) {
    dim[0] = ni; dim[1] = nj;
    data.resize(dim[0]*dim[1]);
    func(data, x, ni, nj, i_begin, j_begin);
  }

  Dense::Dense(const Dense& A) : Node(A.i_abs,A.j_abs,A.level), data(A.data) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1];
  }

  Dense::Dense(Dense&& A) {
    swap(*this, A);
  }

  Dense::Dense(const Dense* A) : Node(A->i_abs,A->j_abs,A->level), data(A->data) {
    dim[0]=A->dim[0]; dim[1]=A->dim[1];
  }

  Dense::Dense(const Block& A)
  : Node((*A.ptr).i_abs, (*A.ptr).j_abs, (*A.ptr).level) {
    assert(A.is(HICMA_DENSE));
    Dense& ref = static_cast<Dense&>(*A.ptr);
    dim[0]=ref.dim[0]; dim[1]=ref.dim[1];
    data = ref.data;
  }

  Dense* Dense::clone() const {
    return new Dense(*this);
  }

  void swap(Dense& first, Dense& second) {
    using std::swap;
    swap(first.data, second.data);
    swap(first.dim, second.dim);
    swap(first.i_abs, second.i_abs);
    swap(first.j_abs, second.j_abs);
    swap(first.level, second.level);
  }

  const Node& Dense::operator=(const double a) {
    for (int i=0; i<dim[0]*dim[1]; i++)
      data[i] = a;
    return *this;
  }

  const Node& Dense::operator=(const Node& A) {
    if (A.is(HICMA_DENSE)) {
      const Dense& AR = static_cast<const Dense&>(A);
      dim[0] = AR.dim[0]; dim[1] = AR.dim[1];
      data.resize(dim[0]*dim[1]);
      data = AR.data;
      return *this;
    } else {
      std::cout << this->is_string() << " = " << A.is_string();
      std::cout << " not implemented!" << std::endl;
      return *this;
    }
  };

  const Node& Dense::operator=(Node&& A) {
    if (A.is(HICMA_DENSE)) {
      swap(*this, static_cast<Dense&>(A));
      return *this;
    } else {
      std::cout << this->is_string() << " = " << A.is_string();
      std::cout << " not implemented!" << std::endl;
      return *this;
    }
  };

  const Dense& Dense::operator=(Dense A) {
    swap(*this, A);
    return *this;
  }

  const Node& Dense::operator=(Block A) {
    return *this = std::move(*A.ptr);
  }

  Dense Dense::operator-() const {
    Dense D(dim[0],dim[1]);
    for (int i=0; i<dim[0]*dim[1]; i++) D[i] = -data[i];
    return D;
  }

  Block Dense::operator+(const Node& B) const {
    Block Out(*this);
    Out += B;
    return Out;
  }
  Block Dense::operator+(Block&& B) const {
    return *this + *B.ptr;
  }
  const Node& Dense::operator+=(const Node& B) {
    if (B.is(HICMA_LOWRANK)) {
      const LowRank& BR = static_cast<const LowRank&>(B);
      assert(dim[0] == BR.dim[0] && dim[1] == BR.dim[1]);
      return *this += BR.dense();
    } else if (B.is(HICMA_DENSE)) {
      const Dense& BR = static_cast<const Dense&>(B);
      assert(dim[0] == BR.dim[0] && dim[1] == BR.dim[1]);
      for (int i=0; i<dim[0]*dim[1]; i++) {
        (*this)[i] += BR[i];
      }
      return *this;
    } else {
      std::cout << this->is_string() << " + " << B.is_string();
      std::cout << " is undefined!" << std::endl;
      return *this;
    }
  }
  const Node& Dense::operator+=(Block&& B) {
    return *this += *B.ptr;
  }

  Block Dense::operator-(const Node& B) const {
    Block Out(*this);
    Out -= B;
    return Out;
  }
  Block Dense::operator-(Block&& B) const {
    return *this - *B.ptr;
  }
  const Node& Dense::operator-=(const Node& B) {
    if (B.is(HICMA_LOWRANK)) {
      const LowRank& BR = static_cast<const LowRank&>(B);
      assert(dim[0] == BR.dim[0] && dim[1] == BR.dim[1]);
      return *this -= BR.dense();
    } else if (B.is(HICMA_DENSE)) {
      const Dense& BR = static_cast<const Dense&>(B);
      assert(dim[0] == BR.dim[0] && dim[1] == BR.dim[1]);
      for (int i=0; i<dim[0]*dim[1]; i++) {
        (*this)[i] -= BR[i];
      }
      return *this;
    } else {
      std::cout << this->is_string() << " - " << B.is_string();
      std::cout << " is undefined!" << std::endl;
      return *this;
    }
  }
  const Node& Dense::operator-=(Block&& B) {
    return *this -= *B.ptr;
  }

  Block Dense::operator*(const Node& B) const {
    if (B.is(HICMA_LOWRANK)) {
      const LowRank& BR = static_cast<const LowRank&>(B);
      assert(dim[0] == BR.dim[0] && dim[1] == BR.dim[1]);
      LowRank Out(BR);
      Out.U = *this * BR.U;
      return Out;
    } else if (B.is(HICMA_DENSE)) {
      const Dense& BR = static_cast<const Dense&>(B);
      assert(dim[1] == BR.dim[0]);
      Dense Out(dim[0], BR.dim[1]);
      if (BR.dim[1] == 1) {
        cblas_dgemv(
                    CblasRowMajor,
                    CblasNoTrans,
                    dim[0],
                    dim[1],
                    1,
                    &data[0],
                    dim[1],
                    &BR[0],
                    1,
                    0,
                    &Out[0],
                    1
                    );
      }
      else {
        cblas_dgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    Out.dim[0],
                    Out.dim[1],
                    dim[1],
                    1,
                    &data[0],
                    dim[1],
                    &BR[0],
                    BR.dim[1],
                    0,
                    &Out[0],
                    Out.dim[1]
                    );
      }
      return Out;
    } else {
      std::cout << this->is_string() << " * " << B.is_string();
      std::cout << " is undefined!" << std::endl;
      return Block();
    }
  }
  Block Dense::operator*(Block&& B) const {
    return *this * *B.ptr;
  }

  double& Dense::operator[](const int i) {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  const double& Dense::operator[](const int i) const {
    assert(i<dim[0]*dim[1]);
    return data[i];
  }

  double& Dense::operator()(const int i, const int j) {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  const double& Dense::operator()(const int i, const int j) const {
    assert(i<dim[0] && j<dim[1]);
    return data[i*dim[1]+j];
  }

  const bool Dense::is(const int enum_id) const {
    return enum_id == HICMA_DENSE;
  }

  const char* Dense::is_string() const { return "Dense"; }

  void Dense::resize(int i) {
    dim[0]=i; dim[1]=1;
    data.resize(dim[0]*dim[1]);
  }

  void Dense::resize(int i, int j) {
    dim[0]=i; dim[1]=j;
    data.resize(dim[0]*dim[1]);
  }

  double Dense::norm() const {
    double l2 = 0;
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        l2 += data[i*dim[1]+j] * data[i*dim[1]+j];
      }
    }
    return l2;
  }

  void Dense::print() const {
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        std::cout << std::setw(20) << std::setprecision(15) << data[i*dim[1]+j] << ' ';
      }
      std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
  }

  void Dense::getrf() {
    std::vector<int> ipiv(std::min(dim[0],dim[1]));
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim[0], dim[1], &data[0], dim[1], &ipiv[0]);
  }

  void Dense::trsm(const Node& A, const char& uplo) {
    if (A.is(HICMA_DENSE)) {
      const Dense& AR = static_cast<const Dense&>(A);
      if (dim[1] == 1) {
        switch (uplo) {
        case 'l' :
          cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                      dim[0], dim[1], 1, &AR[0], AR.dim[1], &data[0], dim[1]);
          break;
        case 'u' :
          cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                      dim[0], dim[1], 1, &AR[0], AR.dim[1], &data[0], dim[1]);
          break;
        default :
          fprintf(stderr,"Second argument must be 'l' for lower, 'u' for upper.\n"); abort();
        }
      }
      else {
        switch (uplo) {
        case 'l' :
          cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                      dim[0], dim[1], 1, &AR[0], AR.dim[1], &data[0], dim[1]);
          break;
        case 'u' :
          cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                      dim[0], dim[1], 1, &AR[0], AR.dim[1], &data[0], dim[1]);
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

  void Dense::gemm(const Node& A, const Node& B) {
    if (A.is(HICMA_DENSE)) {
      if (B.is(HICMA_DENSE)) {
        *this = *this - (A * B);
      } else if (B.is(HICMA_LOWRANK)) {
        *this = *this - (A * B);
      } else if (B.is(HICMA_HIERARCHICAL)) {
        fprintf(
            stderr,"%s -= %s * %s undefined.\n",
            this->is_string(), A.is_string(), B.is_string());
        abort();
      }
    } else if (A.is(HICMA_LOWRANK)) {
      if (B.is(HICMA_DENSE)) {
        *this = *this - (A * B);
      } else if (B.is(HICMA_LOWRANK)) {
        *this = *this - (A * B);
      } else if (B.is(HICMA_HIERARCHICAL)) {
        fprintf(
            stderr,"%s -= %s * %s undefined.\n",
            this->is_string(), A.is_string(), B.is_string());
        abort();
      }
    } else {
      fprintf(
          stderr,"%s -= %s * %s undefined.\n",
          this->is_string(), A.is_string(), B.is_string());
      abort();
    }
  }
}
