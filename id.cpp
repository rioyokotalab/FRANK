#include "id.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

namespace hicma {

  /* C = A*B */
  void matrix_matrix_mult(
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
                CblasNoTrans,
                CblasNoTrans,
                nrows_a,
                ncols_b,
                nrows_b,
                1,
                &A[0],
                ncols_a,
                &B[0],
                ncols_b,
                1,
                &C[0],
                ncols_b);
  }

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


  /* compute compact QR factorization
   * Bt - input matrix. rank x ncols
   * Q - matrix in which Q is to be saved. ncols x rank
   * R - matrix in which R is to be saved. rank x rank
   M is mxn; Q is mxk and R is kxk
   NOTE: FUNCTION NOT USED AS OF NOW.
  */
  void compute_QR_compact_factorization(
                                        std::vector<double>& Bt,
                                        std::vector<double>& Q,
                                        std::vector<double>& R,
                                        int nrows,
                                        int ncols,
                                        int rank)
  {
    int k = min(ncols, rank);
    std::vector<double> QR_temp(Bt);
    std::vector<double> TAU(k);
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, ncols, rank, &QR_temp[0], rank, &TAU[0]);
    for(int i=0; i<ncols; i++) {
      for(int j=0; j<rank; j++) {
        if(j>=i){
          R[i*rank + j] = QR_temp[i*rank + j];
        }
      }
    }
    for (int i=0; i<k; i++) Q[i*k+i] = 1.0;
    LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'N', ncols, rank, rank, &QR_temp[0], rank, &TAU[0], &Q[0], rank);
  }


  /* compute compact QR factoriation and get Q
   * M - original matrix. dim: nrows * rank
   * Q - Q part of QR factorized matrix. Answer is returned in this vector. nrows*k. no data during input.
   * nrows - number of rows of the Matrix Q.
   * ncols - number of columns of M.
   * rank - target rank.
   */
  void QR_factorization_getQ(std::vector<double>& M, std::vector<double>& Q, int nrows, int ncols, int rank){
    int k = min(nrows, rank);
    std::vector<double> QR_temp(M);
    std::vector<double> TAU(k);
    for (int i=0; i<k; i++) Q[i*k+i] = 1.0;
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, nrows, rank, &QR_temp[0], rank, &TAU[0]);
    LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'N', nrows, rank, rank, &QR_temp[0], rank, &TAU[0], &Q[0], rank);
  }

  /* build diagonal matrix from vector elements */
  void build_diagonal_matrix(std::vector<double>& dvals, int n, std::vector<double>& D){
    for(int i=0; i<n; i++){
      D[i*n+i] = dvals[i];
    }
  }

  /* P = U*S*V^T */
  void form_svd_product_matrix(
                               std::vector<double>& U,
                               std::vector<double>& S,
                               std::vector<double>& V,
                               std::vector<double>& P,
                               int nrows,
                               int ncols,
                               int rank
                               ) {
    std::vector<double> US(nrows*rank);
    matrix_matrix_mult(U, S, US, nrows, rank, rank, rank);
    matrix_matrix_mult(US, V, P, nrows, rank, rank, ncols);
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
    std::vector<double> SU(min(nrows, ncols) - 1);
    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', rank, rank, &M[0], rank, &S[0], &U[0], rank, &Vt[0], rank, &SU[0]);
  }

  void transpose(std::vector<double>&  mat, std::vector<double>& mat_t, int nrows, int ncols)
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
    Dense RN(ncols, rank);
    std::mt19937 generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i=0; i<ncols*rank; i++) {
      RN[i] = distribution(generator);
    }

    // Y = M * RN
    std::vector<double> Y(nrows*rank);
    matrix_matrix_mult(A.data, RN.data, Y, nrows, ncols, ncols, rank);

    // [Q, R] = qr(Y)
    std::vector<double> Q(nrows*rank);
    QR_factorization_getQ(Y, Q, nrows, ncols, rank);

    // B' = M' * Q
    std::vector<double> Bt(ncols*rank);
    matrix_transpose_matrix_mult(A.data, Q, Bt, nrows, ncols, nrows, rank);

    // [Qhat, Rhat] = qr(Bt)
    std::vector<double> Qhat(ncols*rank);
    std::vector<double> Rhat(rank*rank);
    compute_QR_compact_factorization(Bt, Qhat, Rhat, nrows, ncols, rank);

    // [Uhat, S, Vhat] = svd(Rhat);
    std::vector<double> Uhat(rank*rank);
    std::vector<double> Shat(rank);
    std::vector<double> Vhat(rank*rank);
    calculate_svd(Uhat, Shat, Vhat, Rhat, rank, rank, rank);
    build_diagonal_matrix(Shat, rank, S.data);

    // Vhat = Vhat'
    std::vector<double> Vhat_t(rank*rank);
    transpose(Vhat, Vhat_t, rank, rank);

    // U = Q * Vhat'
    matrix_matrix_mult(Q, Vhat_t, U.data, nrows, rank, rank, rank);

    // V' = Qhat * Uhat
    std::vector<double> V_t(ncols*rank);
    matrix_matrix_mult(Qhat, Uhat, V_t, ncols, rank, rank, rank);

    // V = V'
    transpose(V_t, V.data, rank, ncols);
  }
}
