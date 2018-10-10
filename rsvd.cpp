#include "rsvd.h"

namespace hicma {

  /* C = A^T*B
   *
   * A - nrows_a * ncols_a
   * B - nrows_b * ncols_b
   * C - nrows_a * ncols_b
   */
  void matrix_transpose_matrix_mult(
                                    const std::vector<double>& A,
                                    std::vector<double>& B,
                                    std::vector<double>& C,
                                    int nrows_a,
                                    int ncols_a,
                                    int nrows_b,
                                    int ncols_b
                                    ) {
    cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                ncols_a,
                ncols_b,
                nrows_a,
                1,
                &A[0],
                ncols_a,
                &B[0],
                ncols_b,
                1,
                &C[0],
                ncols_b
                );
  }

  /* build diagonal matrix from vector elements */
  void build_diagonal_matrix(std::vector<double>& dvals, int n, std::vector<double>& D){
    for(int i=0; i<n; i++){
      D[i*n+i] = dvals[i];
    }
  }

  /*
   * Outputs:
   * U -> rank*rank
   * S -> rank*rank
   * Vt -> rank*rank
   *
   * Inputs:
   * M -> nrows*ncols.
   */
  void calculate_svd(
                     std::vector<double>& U,
                     std::vector<double>& S,
                     std::vector<double>& Vt,
                     std::vector<double>& M,
                     int nrows,
                     int ncols,
                     int rank
                     ) {
    std::vector<double> SU(std::min(nrows, ncols) - 1);
    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', rank, rank, &M[0], rank, &S[0], &U[0], rank, &Vt[0], rank, &SU[0]);
  }

  void transpose(const std::vector<double>&  mat, std::vector<double>& mat_t, int nrows, int ncols)
  {
    for (int i=0; i<nrows; i++) {
      for (int j=0; j<ncols; j++) {
        mat_t[i*ncols+j] = mat[j*nrows+i];
      }
    }
  }

  /* Randomized low rank SVD of matrix A
   * A - input matrix.
   * rank - rank of the output.
   * U - U matrix from RSVD.
   * S - S matrix from RSVD.
   * V - V matrix from RSVD.
   */
  void rsvd(
            const Dense& A,
            int rank,
            Dense& U,
            Dense& S,
            Dense& V
            ) {
    int nrows = A.dim[0];
    int ncols = A.dim[1];

    // RN = randn(n,k+p)
    Dense RN(ncols,rank);
    std::mt19937 generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i=0; i<ncols*rank; i++) {
      RN[i] = distribution(generator);
    }

    // Y = A * RN
    Dense Y(nrows,rank);
    Y.gemm(A, RN, 1, 0);

    // [Q, R] = qr(Y)
    Dense Q(nrows,rank);
    Dense R(rank,rank);
    Y.qr(Q, R);

    // B' = A' * Q
    Dense At(ncols,nrows);
    transpose(A.data, At.data, ncols, nrows);

    Dense Bt(ncols,rank);
    Bt.gemm(At, Q, 1, 0);

    // [Qhat, Rhat] = qr(B')
    Dense Qhat(ncols,rank);
    Dense Rhat(rank,rank);
    Bt.qr(Qhat,Rhat);

    // [Uhat, S, Vhat] = svd(Rhat);
    Dense Uhat(rank,rank);
    Dense Vhat(rank,rank);
    Rhat.svd(Uhat,S,Vhat);

    // Vhat = Vhat'
    Dense Vhat_t(rank,rank);
    transpose(Vhat.data, Vhat_t.data, rank, rank);

    // U = Q * Vhat'
    U.gemm(Q, Vhat_t);

    // V' = Qhat * Uhat
    Dense V_t(ncols,rank);
    V_t.gemm(Qhat, Uhat);

    // V = V'
    transpose(V_t.data, V.data, rank, ncols);
  }
}
