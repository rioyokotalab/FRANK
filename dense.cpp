#include "dense.h"
#include <lapacke.h>
#include "low_rank.h"
#include "hierarchical.h"
#include <iomanip>

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

  Dense::Dense(const Dense& A) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1];
    data.resize(dim[0]*dim[1]);
    data = A.data;
  }

  Dense::Dense(
      const Hierarchical* parent,
      const int i_rel,
      const int j_rel,
      const size_t xi_half,
      const size_t xj_half) : Node(parent, i_rel, j_rel) {

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

  const Dense& Dense::operator=(const Dense A) {
    dim[0]=A.dim[0]; dim[1]=A.dim[1];
    data.resize(dim[0]*dim[1]);
    data = A.data;
    return *this;
  }

  const Dense Dense::operator+=(const Dense& A) {
    for (int i=0; i<dim[0]*dim[1]; i++)
      this->data[i] += A.data[i];
    return *this;
  }

  const Dense Dense::operator-=(const Dense& A) {
    for (int i=0; i<dim[0]*dim[1]; i++)
      this->data[i] -= A.data[i];
    return *this;
  }

  Dense Dense::operator+(const Dense& A) const {
    return Dense(*this) += A;
  }

  Dense Dense::operator-(const Dense& A) const {
    return Dense(*this) -= A;
  }

  Dense Dense::operator*(const Dense& B) const {
    Dense C(dim[0],B.dim[1]);
    if (B.dim[1] == 1) {
      cblas_dgemv(CblasRowMajor, CblasNoTrans, dim[0], dim[1], 1, &data[0], dim[1],
                  &B[0], 1, 0, &C[0], 1);
    }
    else {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.dim[0], C.dim[1], dim[1], 1,
                  &data[0], dim[1], &B[0], B.dim[1], 0, &C[0], C.dim[1]);
    }
    return C;
  }

  LowRank Dense::operator*(LowRank& A) {
    return LowRank((*this) * A.dense(), A.rank);
  }

  Dense Dense::operator-() const {
    Dense D(dim[0],dim[1]);
    for (int i=0; i<dim[0]*dim[1]; i++) D[i] = -data[i];
    return D;
  }

  std::vector<int> Dense::getrf() {
    std::vector<int> ipiv(std::min(dim[0],dim[1]));
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, dim[0], dim[1], &data[0], dim[1], &ipiv[0]);
    return ipiv;
  }

  void Dense::trsm(Dense& A, const char& uplo) {
    if (dim[1] == 1) {
      switch (uplo) {
      case 'l' :
        cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    dim[0], dim[1], 1, &A[0], A.dim[1], &data[0], dim[1]);
        break;
      case 'u' :
        cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                    dim[0], dim[1], 1, &A[0], A.dim[1], &data[0], dim[1]);
        break;
      default :
        fprintf(stderr,"Second argument must be 'l' for lower, 'u' for upper.\n");
      }
    }
    else {
      switch (uplo) {
      case 'l' :
        cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    dim[0], dim[1], 1, &A[0], A.dim[1], &data[0], dim[1]);
        break;
      case 'u' :
        cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                    dim[0], dim[1], 1, &A[0], A.dim[1], &data[0], dim[1]);
        break;
      default :
        fprintf(stderr,"Second argument must be 'l' for lower, 'u' for upper.\n");
      }
    }
  }

  void Dense::gemv(const Dense& A, const Dense& b) {
    *this -= A * b;
  }

  void Dense::gemv(const LowRank& A, const Dense& b) {
    *this -= A.U * A.B * (A.V * b);
  }

  void Dense::gemm(const Dense& A, const Dense& B) {
    *this -= A * B;
  }

  void Dense::gemm(const Dense& A, const LowRank& B) {
    *this -= ((A * B.U) * B.B) * B.V;
  }

  void Dense::gemm(const LowRank& A, const Dense& B) {
    *this -= A.U * (A.B * (A.V * B));
  }

  void Dense::gemm(const LowRank& A, const LowRank& B) {
    *this -= A.U * (A.B * (A.V * B.U) * B.B) * B.V;
  }

  void Dense::resize(int i) {
    dim[0]=i; dim[1]=1;
    data.resize(dim[0]*dim[1]);
  }

  void Dense::resize(int i, int j) {
    dim[0]=i; dim[1]=j;
    data.resize(dim[0]*dim[1]);
  }

  double Dense::norm() {
    double l2 = 0;
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        l2 += data[i*dim[1]+j] * data[i*dim[1]+j];
      }
    }
    return l2;
  }

  void Dense::print() {
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        std::cout << std::setw(20) << std::setprecision(15) << data[i*dim[1]+j] << ' ';
      }
      std::cout << std::endl;
    }
  }
}
