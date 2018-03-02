#ifndef dense_h
#define dense_h
#include <cassert>
#include "node.h"
#include <vector>

extern "C" {
  void dgetrf_(const int* M, const int* N, const double* A, const int* LDA, const int* IPIV, const int* INFO);
  void dtrsm_(const char* SIDE, const char* UPLO, const char* TRANSA, const char* DIAG, const int* M, const int* N, const double* ALPHA, const double* A, const int* LDA, const double* B, const int* LDB);
  void dgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K, const double* ALPHA, const double* A, const int* LDA, const double* B, const int* LDB, const double* BETA, const double* C, const int* LDC);
  void dgemv_(const char* TRANS, const int* M, const int* N, const double* ALPHA, const double* A, const int* LDA, const double* X, const int* INCX, const double* BETA, const double* Y, const int* INCY);
}

namespace hicma {
  class Dense : public Node {
  public:
    std::vector<double> data;
    int dim[2];

    Dense() {
      dim[0]=0; dim[1]=0;
    }

    Dense(int i) {
      dim[0]=i; dim[1]=1;
      data.resize(dim[0]*dim[1]);
    }

    Dense(int i, int j) {
      dim[0]=i; dim[1]=j;
      data.resize(dim[0]*dim[1]);
    }

    double& operator[](const int i) {
      return data[i];
    }

    const double& operator[](const int i) const {
      return data[i];
    }

    double& operator()(const int i, const int j) {
      assert(i<dim[0] && j<dim[1]);
      return data[i*dim[1]+j];
    }

    const double& operator()(const int i, const int j) const {
      assert(i<dim[0] && j<dim[1]);
      return data[i*dim[1]+j];
    }

    const Dense operator+=(const Dense& D) {
      for (int i=0; i<dim[0]*dim[1]; i++)
        this->data[i] += D.data[i];
      return *this;
    }

    Dense operator+(const Dense& D) const {
      return Dense(*this) += D;
    }

    Dense operator*(const Dense& B) const {
      char c_n='n';
      double zero = 0;
      double one = 1;
      Dense C(dim[0], B.dim[1]);
      dgemm_(&c_n, &c_n, &dim[0], &B.dim[1], &dim[1], &zero, &data[0], &dim[0], &B[0], &B.dim[0], &one, &C[0], &C.dim[0]);
      return C;
    }

    void getrf(std::vector<int>& ipiv) const {
      int info;
      dgetrf_(&dim[0], &dim[1], &data[0], &dim[0], &ipiv[0], &info);
    }

    void trsm(Dense& D, const char& uplo) const {
      double one = 1;
      char c_l='l', c_r='r', c_u='u', c_n='n', c_t='t';
      if (dim[1] == 1) {
        switch (uplo) {
        case 'l' :
          dtrsm_(&c_l, &c_l, &c_n, &c_u, &dim[0], &dim[1], &one, &D[0], &dim[0], &data[0], &dim[0]);
          break;
        case 'u' :
          dtrsm_(&c_l, &c_u, &c_n, &c_n, &dim[0], &dim[1], &one, &D[0], &dim[0], &data[0], &dim[0]);
          break;
        default :
          fprintf(stderr,"First argument must be 'l' for lower, 'u' for upper.\n");
        }
      } else {
        switch (uplo) {
        case 'l' :
          dtrsm_(&c_r, &c_l, &c_t, &c_u, &dim[0], &dim[1], &one, &D[0], &dim[1], &data[0], &dim[0]);
          break;
        case 'u' :
          dtrsm_(&c_l, &c_u, &c_t, &c_n, &dim[0], &dim[1], &one, &D[0], &dim[0], &data[0], &dim[0]);
          break;
        default :
          fprintf(stderr,"First argument must be 'l' for lower, 'u' for upper.\n");
        }
      }
    }

    void gemv(Dense& A, Dense& b) const {
      assert(dim[1] == 1 && b.dim[1] == 1);
      char c_t='t';
      int i1 = 1;
      double p1 = 1;
      double m1 = -1;
      dgemv_(&c_t, &A.dim[0], &A.dim[1], &m1, &A[0], &A.dim[0], &b[0], &i1, &p1, &data[0], &i1);
    }

    void gemm(Dense& A, Dense& B) const {
      assert(dim[0] == A.dim[0] && dim[1] == B.dim[1] && A.dim[1] == B.dim[0]);
      char c_n='n';
      double p1 = 1;
      double m1 = -1;
      dgemm_(&c_n, &c_n, &dim[0], &dim[1], &A.dim[1], &m1, &A[0], &A.dim[0], &B[0], &B.dim[0], &p1, &data[0], &dim[0]);
    }

    void resize(int i) {
      dim[0]=i; dim[1]=1;
      data.resize(dim[0]*dim[1]);
    }

    void resize(int i, int j) {
      dim[0]=i; dim[1]=j;
      data.resize(dim[0]*dim[1]);
    }

    void print() {
      for (int i=0; i<dim[0]; i++) {
        for (int j=0; j<dim[1]; j++) {
          std::cout << data[dim[0]*i+j] << ' ';
        }
        std::cout << std::endl;
      }
    }
  };
}
#endif
