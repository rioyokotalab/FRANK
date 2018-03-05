#include "dense.h"
#include "low_rank.h"

extern "C" {
  void dgetrf_(const int* M, const int* N, const double* A, const int* LDA, const int* IPIV, const int* INFO);
  void dtrsm_(const char* SIDE, const char* UPLO, const char* TRANSA, const char* DIAG, const int* M, const int* N, const double* ALPHA, const double* A, const int* LDA, const double* B, const int* LDB);
  void dgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K, const double* ALPHA, const double* A, const int* LDA, const double* B, const int* LDB, const double* BETA, const double* C, const int* LDC);
  void dgemv_(const char* TRANS, const int* M, const int* N, const double* ALPHA, const double* A, const int* LDA, const double* X, const int* INCX, const double* BETA, const double* Y, const int* INCY);
}

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

  Dense Dense::operator*(const Dense& A) const {
    char c_n='n';
    double zero = 1;
    double one = 1;
    Dense C(dim[0], A.dim[1]);
    dgemm_(&c_n, &c_n, &A.dim[1], &dim[0], &dim[1], &one, &A[0], &A.dim[1], &data[0], &dim[1], &zero, &C[0], &C.dim[1]);
    return C;
  }

  LowRank Dense::operator*(LowRank& A) {
    return LowRank((*this) * (A.U * A.B * A.V), A.rank);
  }

  std::vector<int> Dense::getrf() const {
    int info;
    std::vector<int> ipiv(std::min(dim[0],dim[1]));
    dgetrf_(&dim[0], &dim[1], &data[0], &dim[0], &ipiv[0], &info);
    return ipiv;
  }

  void Dense::trsm(Dense& A, const char& uplo) const {
    double one = 1;
    char c_l='l', c_r='r', c_u='u', c_n='n', c_t='t';
    if (dim[1] == 1) {
      switch (uplo) {
      case 'l' :
        dtrsm_(&c_l, &c_l, &c_n, &c_u, &dim[0], &dim[1], &one, &A[0], &dim[0], &data[0], &dim[0]);
        break;
      case 'u' :
        dtrsm_(&c_l, &c_u, &c_n, &c_n, &dim[0], &dim[1], &one, &A[0], &dim[0], &data[0], &dim[0]);
        break;
      default :
        fprintf(stderr,"Second argument must be 'l' for lower, 'u' for upper.\n");
      }
    } else {
      switch (uplo) {
      case 'l' :
        dtrsm_(&c_r, &c_l, &c_t, &c_u, &dim[0], &dim[1], &one, &A[0], &A.dim[1], &data[0], &dim[0]);
        break;
      case 'u' :
        dtrsm_(&c_l, &c_u, &c_t, &c_n, &dim[0], &dim[1], &one, &A[0], &A.dim[0], &data[0], &dim[0]);
        break;
      default :
        fprintf(stderr,"Second argument must be 'l' for lower, 'u' for upper.\n");
      }
    }
  }

  void Dense::gemv(Dense& A, Dense& b) const {
    assert(dim[1] == 1 && b.dim[1] == 1);
    char c_t='t';
    int i1 = 1;
    double p1 = 1;
    double m1 = -1;
    dgemv_(&c_t, &A.dim[0], &A.dim[1], &m1, &A[0], &A.dim[0], &b[0], &i1, &p1, &data[0], &i1);
  }

  void Dense::gemm(Dense& A, Dense& B) const {
    assert(dim[0] == B.dim[0] && dim[1] == A.dim[1] && B.dim[1] == A.dim[0]);
    char c_n='n';
    double p1 = 1;
    double m1 = -1;
    dgemm_(&c_n, &c_n, &dim[0], &dim[1], &B.dim[1], &m1, &B[0], &B.dim[0], &A[0], &A.dim[0], &p1, &data[0], &dim[0]);
  }

  void Dense::gemm(Dense& A, LowRank& B) const {
    assert(dim[0] == B.dim[0] && dim[1] == A.dim[1] && B.dim[1] == A.dim[0]);
    char c_n='n';
    double p1 = 1;
    double m1 = -1;
    Dense BD = B.dense();
    dgemm_(&c_n, &c_n, &dim[0], &dim[1], &BD.dim[1], &m1, &BD[0], &BD.dim[0], &A[0], &A.dim[0], &p1, &data[0], &dim[0]);
  }

  void Dense::gemm(LowRank& A, Dense& B) const {
    assert(dim[0] == B.dim[0] && dim[1] == A.dim[1] && B.dim[1] == A.dim[0]);
    char c_n='n';
    double p1 = 1;
    double m1 = -1;
    Dense AD = A.dense();
    dgemm_(&c_n, &c_n, &dim[0], &dim[1], &B.dim[1], &m1, &B[0], &B.dim[0], &AD[0], &AD.dim[0], &p1, &data[0], &dim[0]);
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
        std::cout << data[i*dim[1]+j] << ' ';
      }
      std::cout << std::endl;
    }
  }
}
