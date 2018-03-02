#ifndef low_rank_h
#define low_rank_h
#include <cassert>
#include "dense.h"
#include "id.h"
#include "node.h"
#include <vector>

namespace hicma {
  class LowRank : public Node {
  public:
    Dense U, B, V;
    int dim[2];
    int rank;

    LowRank() {
      dim[0]=0; dim[1]=0; rank=0;
    }

    LowRank(int i, int j, int k) {
      dim[0]=i; dim[1]=j; rank=k;
      U.resize(dim[0],rank);
      B.resize(rank,rank);
      V.resize(rank,dim[1]);
    }

    LowRank(LowRank &A) {
      dim[0]=A.dim[0]; dim[1]=A.dim[1]; rank=A.rank;
      for (int i=0; i<dim[0]*rank; i++) U[i] = A.U[i];
      for (int i=0; i<rank*rank; i++) B[i] = A.B[i];
      for (int i=0; i<rank*dim[1]; i++) V[i] = A.V[i];
    }

    LowRank(Dense &D, int k) {
      int m = dim[0] = D.dim[0];
      int n = dim[1] = D.dim[1];
      rank = k;
      gsl_matrix *D2 = gsl_matrix_calloc(m,n);
      gsl_matrix *U2 = gsl_matrix_calloc(m,k);
      gsl_matrix *B2 = gsl_matrix_calloc(k,k);
      gsl_matrix *V2 = gsl_matrix_calloc(n,k);
      for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
          D2->data[i*n+j] = D(i,j);
        }
      }
      gsl_matrix *RN = gsl_matrix_calloc(n,k);
      initialize_random_matrix(RN);
      gsl_matrix *Y = gsl_matrix_alloc(m,k);
      matrix_matrix_mult(D2, RN, Y);
      gsl_matrix *Q = gsl_matrix_alloc(m,k);
      QR_factorization_getQ(Y, Q);
      gsl_matrix *Bt = gsl_matrix_alloc(n,k);
      matrix_transpose_matrix_mult(D2,Q,Bt);
      gsl_matrix *Qhat = gsl_matrix_calloc(n,k);
      gsl_matrix *Rhat = gsl_matrix_calloc(k,k);
      compute_QR_compact_factorization(Bt,Qhat,Rhat);
      gsl_matrix *Uhat = gsl_matrix_alloc(k,k);
      gsl_vector *Sigmahat = gsl_vector_alloc(k);
      gsl_matrix *Vhat = gsl_matrix_alloc(k,k);
      gsl_vector *svd_work_vec = gsl_vector_alloc(k);
      gsl_matrix_memcpy(Uhat, Rhat);
      gsl_linalg_SV_decomp (Uhat, Vhat, Sigmahat, svd_work_vec);
      build_diagonal_matrix(Sigmahat, k, B2);
      matrix_matrix_mult(Q,Vhat,U2);
      matrix_matrix_mult(Qhat,Uhat,V2);
      for(int i=0; i<m; i++){
        for(int j=0; j<k; j++){
          U(i,j) = U2->data[i*k+j];
        }
      }
      for(int i=0; i<k; i++){
        for(int j=0; j<k; j++){
          B(i,j) = B2->data[i*k+j];
        }
      }
      for(int i=0; i<n; i++){
        for(int j=0; j<k; j++){
          V(j,i) = V2->data[i*k+j];
        }
      }
      gsl_matrix_free(D2);
      gsl_matrix_free(U2);
      gsl_matrix_free(B2);
      gsl_matrix_free(V2);
      gsl_matrix_free(Y);
      gsl_matrix_free(Q);
      gsl_matrix_free(Rhat);
      gsl_matrix_free(Qhat);
      gsl_matrix_free(Uhat);
      gsl_matrix_free(Vhat);
      gsl_matrix_free(Bt);
    }

    Dense operator+(const Dense& D) const {
      return U * B * V + D;
    }

    LowRank operator*(const Dense& D) {
      LowRank A = *this;
      A.V = A.V * D;
      return A;
    }

  };
}
#endif
