#include "hicma/classes/dense.h"

#include "hicma/classes/node.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <memory>
#include <utility>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include "yorel/multi_methods.hpp"

namespace hicma {

  Dense::Dense() {
    MM_INIT();
    dim[0]=0; dim[1]=0;
  }

  Dense::Dense(const int m) {
    MM_INIT();
    dim[0]=m; dim[1]=1; data.resize(dim[0],0);
  }

  Dense::Dense(
    const int m, const int n,
    const int i_abs, const int j_abs,
    const int level
  ) : Node(i_abs, j_abs, level) {
    MM_INIT();
    dim[0]=m; dim[1]=n; data.resize(dim[0]*dim[1],0);
  }

  Dense::Dense(
    void (*func)(
      std::vector<double>& data,
      std::vector<double>& x,
      const int& ni, const int& nj,
      const int& i_begin, const int& j_begin
    ),
    std::vector<double>& x,
    const int ni, const int nj,
    const int i_begin, const int j_begin,
    const int i_abs, const int j_abs,
    const int level
  ) : Node(i_abs, j_abs, level) {
    MM_INIT();
    dim[0] = ni; dim[1] = nj;
    data.resize(dim[0]*dim[1]);
    func(data, x, ni, nj, i_begin, j_begin);
  }

  Dense::Dense(
    void (*func)(
      std::vector<double>& data,
      std::vector<std::vector<double>>& x,
      const int& ni, const int& nj,
      const int& i_begin, const int& j_begin
    ),
    std::vector<std::vector<double>>& x,
    const int ni, const int nj,
    const int i_begin, const int j_begin,
    const int i_abs, const int j_abs,
    const int level
  ) : Node(i_abs,j_abs,level) {
    MM_INIT();
    dim[0] = ni; dim[1] = nj;
    data.resize(dim[0]*dim[1]);
    func(data, x, ni, nj, i_begin, j_begin);
  }

  Dense::Dense(const Dense& A) : Node(A.i_abs, A.j_abs, A.level), data(A.data) {
    MM_INIT();
    dim[0] = A.dim[0]; dim[1] = A.dim[1];
  }

  Dense::Dense(Dense&& A) {
    MM_INIT();
    swap(*this, A);
  }

  Dense::Dense(const LowRank& A) : Node(A.i_abs, A.j_abs, A.level) {
    MM_INIT();
    dim[0] = A.dim[0]; dim[1] = A.dim[1];
    data.resize(dim[0]*dim[1], 0);
    Dense UxS(A.dim[0], A.rank);
    gemm(A.U, A.S, UxS, 1, 0);
    gemm(UxS, A.V, *this, 1, 0);
  }

  Dense::Dense(const Hierarchical& A) : Node(A.i_abs, A.j_abs, A.level) {
    MM_INIT();
    dim[0] = get_n_rows(A);
    dim[1] = get_n_cols(A);
    data.resize(dim[0]*dim[1]);
    int i_begin = 0;
    for (int i=0; i<A.dim[0]; i++) {
      int j_begin = 0;
      for (int j=0; j<A.dim[1]; j++) {
        Dense AD = Dense(A(i,j));
        for (int ic=0; ic<AD.dim[0]; ic++) {
          for (int jc=0; jc<AD.dim[1]; jc++) {
            (*this)(ic+i_begin, jc+j_begin) = AD(ic,jc);
          }
        }
        j_begin += AD.dim[1];
      }
      i_begin += get_n_rows(A(i, 0));
    }
  }

  Dense::Dense(const NodeProxy& A) : Node(A->i_abs, A->j_abs, A->level) {
    MM_INIT();
    *this = make_dense(A);
  }

  std::unique_ptr<Node> Dense::clone() const {
    return std::make_unique<Dense>(*this);
  }

  std::unique_ptr<Node> Dense::move_clone() {
    return std::make_unique<Dense>(std::move(*this));
  }

  void swap(Dense& A, Dense& B) {
    using std::swap;
    swap(static_cast<Node&>(A), static_cast<Node&>(B));
    swap(A.data, B.data);
    swap(A.dim, B.dim);
  }

  const Dense& Dense::operator=(const double a) {
    for (int i=0; i<dim[0]*dim[1]; i++)
      data[i] = a;
    return *this;
  }

  const Dense& Dense::operator=(Dense A) {
    swap(*this, A);
    return *this;
  }

  Dense Dense::operator+(const Dense& A) const {
    Dense B(*this);
    B += A;
    return B;
  }

  Dense Dense::operator-(const Dense& A) const {
    Dense B(*this);
    B -= A;
    return B;
  }

  const Dense& Dense::operator+=(const Dense& A) {
    assert(dim[0] == A.dim[0]);
    assert(dim[1] == A.dim[1]);
    for (int i=0; i<dim[0]*dim[1]; i++) {
      (*this)[i] += A[i];
    }
    return *this;
  }

  const Dense& Dense::operator-=(const Dense& A) {
    assert(dim[0] == A.dim[0]);
    assert(dim[1] == A.dim[1]);
    for (int i=0; i<dim[0]*dim[1]; i++) {
      (*this)[i] -= A[i];
    }
    return *this;
  }

  const Dense& Dense::operator*=(const double a) {
    for (int i=0; i<dim[0]*dim[1]; i++) {
      (*this)[i] *= a;
    }
    return *this;
  }

  double& Dense::operator[](const int i) {
    assert(i < dim[0]*dim[1]);
    return data[i];
  }

  const double& Dense::operator[](const int i) const {
    assert(i < dim[0]*dim[1]);
    return data[i];
  }

  double& Dense::operator()(const int i, const int j) {
    assert(i < dim[0]);
    assert(j < dim[1]);
    return data[i*dim[1]+j];
  }

  const double& Dense::operator()(const int i, const int j) const {
    assert(i < dim[0]);
    assert(j < dim[1]);
    return data[i*dim[1]+j];
  }

  const char* Dense::type() const { return "Dense"; }

  int Dense::size() const {
    return dim[0] * dim[1];
  }

  void Dense::resize(const int dim0, const int dim1) {
    assert(dim0 <= dim[0]);
    assert(dim1 <= dim[1]);
    for (int i=0; i<dim0; i++) {
      for (int j=0; j<dim1; j++) {
        data[i*dim1+j] = data[i*dim[1]+j];
      }
    }
    dim[0] = dim0;
    dim[1] = dim1;
  }

  Dense Dense::transpose() const {
    Dense A(dim[1], dim[0], i_abs, j_abs, level);
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        A(j,i) = (*this)(i,j);
      }
    }
    return A;
  }

  void Dense::transpose() {
    std::vector<double> _data(data);
    std::swap(dim[0], dim[1]);
    for(int i=0; i<dim[0]; i++) {
      for(int j=0; j<dim[1]; j++) {
        data[i*dim[1]+j] = _data[j*dim[0]+i];
      }
    }
  }

  void Dense::svd(Dense& U, Dense& S, Dense& V) {
    start("-DGESVD");
    Dense Sdiag(std::min(dim[0], dim[1]), 1);
    Dense work(std::min(dim[0], dim[1])-1, 1);
    LAPACKE_dgesvd(
      LAPACK_ROW_MAJOR,
      'A', 'A',
      dim[0], dim[1],
      &data[0], dim[1],
      &Sdiag[0],
      &U[0], U.dim[1],
      &V[0], V.dim[1],
      &work[0]
    );
    for(int i=0; i<std::min(dim[0], dim[1]); i++){
      S(i, i) = Sdiag[i];
    }
    stop("-DGESVD",false);
  }

  void Dense::sdd(Dense& U, Dense& S, Dense& V) {
    start("-DGESDD");
    Dense Sdiag(std::min(dim[0], dim[1]), 1);
    Dense work(std::min(dim[0], dim[1])-1, 1);
    // dgesdd is faster, but makes little/no difference in randomized SVD
    LAPACKE_dgesdd(
      LAPACK_ROW_MAJOR,
      'A',
      dim[0], dim[1],
      &data[0], dim[1],
      &Sdiag[0],
      &U[0], U.dim[1],
      &V[0], V.dim[1]
    );
    for(int i=0; i<std::min(dim[0], dim[1]); i++){
      S(i, i) = Sdiag[i];
    }
    stop("-DGESDD",false);
  }

  void Dense::svd(Dense& Sdiag) {
    start("-DGESVD");
    Dense U(dim[0],dim[1]),V(dim[1],dim[0]);
    Dense work(dim[1]-1,1);
    // Using 'A' is a major waste of time
    LAPACKE_dgesvd(
      LAPACK_ROW_MAJOR,
      'A', 'A',
      dim[0], dim[1],
      &data[0], dim[1],
      &Sdiag[0],
      &U[0], U.dim[0],
      &V[0], V.dim[0],
      &work[0]
    );
    stop("-DGESVD",false);
  }

  BEGIN_SPECIALIZATION(make_dense, Dense, const Hierarchical& A){
    return Dense(A);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const LowRank& A){
    return Dense(A);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const Dense& A){
    return Dense(A);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const Node& A){
    std::cout << "Cannot create Dense from " << A.type() << "!" << std::endl;
    abort();
  } END_SPECIALIZATION;

} // namespace hicma
