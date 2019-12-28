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

  Dense::Dense() : dim{0, 0} { MM_INIT(); }

  Dense::~Dense() = default;

  Dense::Dense(const Dense& A) {
    MM_INIT();
    *this = A;
  }

  Dense& Dense::operator=(const Dense& A) = default;

  Dense::Dense(Dense&& A) {
    MM_INIT();
    *this = std::move(A);
  }

  Dense& Dense::operator=(Dense&& A) = default;

  Dense::Dense(int m) : dim{m, 1} {
    MM_INIT();
    data.resize(dim[0], 0);
  }

  std::unique_ptr<Node> Dense::clone() const {
    return std::make_unique<Dense>(*this);
  }

  std::unique_ptr<Node> Dense::move_clone() {
    return std::make_unique<Dense>(std::move(*this));
  }

  const char* Dense::type() const { return "Dense"; }

  Dense::Dense(const Node& A) : Node(A) {
    MM_INIT();
    // Test performance of make_dense in this case!!
    // Unnecessary copies possible
    *this = make_dense(A);
  }

  Dense::Dense(
    int m, int n,
    int i_abs, int j_abs,
    int level
  ) : Node(i_abs, j_abs, level), dim{m, n} {
    MM_INIT();
    data.resize(dim[0]*dim[1], 0);
  }

  Dense::Dense(
    void (*func)(
      std::vector<double>& data,
      std::vector<double>& x,
      int ni, int nj,
      int i_begin, int j_begin
    ),
    std::vector<double>& x,
    int ni, int nj,
    int i_begin, int j_begin,
    int i_abs, int j_abs,
    int level
  ) : Node(i_abs, j_abs, level), dim{ni, nj}{
    MM_INIT();
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
  ) : Node(i_abs,j_abs,level), dim{ni, nj} {
    MM_INIT();
    data.resize(dim[0]*dim[1]);
    func(data, x, ni, nj, i_begin, j_begin);
  }

  const Dense& Dense::operator=(const double a) {
    for (int i=0; i<dim[0]*dim[1]; i++)
      data[i] = a;
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

  double& Dense::operator[](int i) {
    assert(i < dim[0]*dim[1]);
    return data[i];
  }

  const double& Dense::operator[](int i) const {
    assert(i < dim[0]*dim[1]);
    return data[i];
  }

  double& Dense::operator()(int i, int j) {
    assert(i < dim[0]);
    assert(j < dim[1]);
    return data[i*dim[1]+j];
  }

  const double& Dense::operator()(int i, int j) const {
    assert(i < dim[0]);
    assert(j < dim[1]);
    return data[i*dim[1]+j];
  }

  int Dense::size() const {
    return dim[0] * dim[1];
  }

  void Dense::resize(int dim0, int dim1) {
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
    // TODO Using 'A' is a major waste of time
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
    Dense B(get_n_rows(A), get_n_cols(A));
    // TODO This loop copies the data multiple times
    int i_begin = 0;
    for (int i=0; i<A.dim[0]; i++) {
      int j_begin = 0;
      for (int j=0; j<A.dim[1]; j++) {
        Dense AD = Dense(A(i,j));
        for (int ic=0; ic<AD.dim[0]; ic++) {
          for (int jc=0; jc<AD.dim[1]; jc++) {
            B(ic+i_begin, jc+j_begin) = AD(ic,jc);
          }
        }
        j_begin += AD.dim[1];
      }
      i_begin += get_n_rows(A(i, 0));
    }
    // TODO Consider return with std::move. Test if the copy is elided!!
    return B;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const LowRank& A){
    Dense B(A.dim[0], A.dim[1]);
    Dense UxS(A.dim[0], A.rank);
    gemm(A.U, A.S, UxS, 1, 0);
    gemm(UxS, A.V, B, 1, 0);
    // TODO Consider return with std::move. Test if the copy is elided!!
    return B;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const Dense& A){
    // TODO Consider return with std::move. Test if the copy is elided!!
    return Dense(A);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const Node& A){
    std::cout << "Cannot create Dense from " << A.type() << "!" << std::endl;
    abort();
  } END_SPECIALIZATION;

} // namespace hicma
