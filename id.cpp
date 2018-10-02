#include "id.h"

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <lapacke.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))
//using namespace std;
namespace hicma {

  void initialize_random_matrix(std::vector<double>& M, int nrows, int ncols)
  {
    boost::mt19937 rng;
    boost::normal_distribution<> nd(0.0, 1.0);
    boost::variate_generator<boost::mt19937&,
                             boost::normal_distribution<> > var_nor(rng, nd);

    for(int i=0; i<nrows*ncols; i++){
      M[i] = var_nor();
    }
  }

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

  /* computes the approximate low rank SVD of rank k of matrix M using QR method */
  /*
   * M - input matrix.
   * k - rank of the output.
   * U - U matrix from ID.
   * S - S (sigma) matrix from ID containing eigenvalues.
   * V - V matrix from ID.
   * nrows - number of rows of M.
   * ncols - number of cols of M.
   */
  void randomized_low_rank_svd2(
                                const std::vector<double>& M,
                                int rank,
                                std::vector<double>& U,
                                std::vector<double>& S,
                                std::vector<double>& V,
                                int nrows,
                                int ncols
                                ) {
    // RN = randn(n,k+p)
    // build random matrix
    std::vector<double> RN(ncols*rank);
    initialize_random_matrix(RN, ncols, rank);

    // Y = M * RN
    // multiply to get matrix of random samples Y
    std::vector<double> Y(nrows*rank);
    matrix_matrix_mult(M, RN, Y, nrows, ncols, ncols, rank);

    // [q, r] = qr(Y,0)
    std::vector<double> Q(nrows*rank);
    QR_factorization_getQ(Y, Q, nrows, ncols, rank);

    // bt = M' * q;
    // form Bt = Qt*M : rankxnrows * nrowsxncols = rankxncols
    std::vector<double> Bt(ncols*rank);
    matrix_transpose_matrix_mult(M, Q, Bt, nrows, ncols, nrows, rank);

    /* // Bt -> ncols * rank, Qhat -> ncols * rank, Rhat -> rank, rank */
    // [Qhat, Rhat] = qr(bt)
    std::vector<double> Qhat(ncols*rank);
    std::vector<double> Rhat(rank*rank);
    compute_QR_compact_factorization(Bt, Qhat, Rhat, nrows, ncols, rank);

    // compute SVD of Rhat
    // [Uhat, S, Vhat] = svd(Rhat);
    std::vector<double> Uhat(rank*rank);
    std::vector<double> Shat(rank);
    std::vector<double> Vhat(rank*rank);
    calculate_svd(Uhat, Shat, Vhat, Rhat, rank, rank, rank);
    build_diagonal_matrix(Shat, rank, S);

    // Vhat = Vhat'
    std::vector<double> Vhat_t(rank*rank);
    transpose(Vhat, Vhat_t, rank, rank);

    // U = q * Vhat
    matrix_matrix_mult(Q, Vhat_t, U, nrows, rank, rank, rank);

    // V = Qhat*Uhat
    std::vector<double> V_t(ncols*rank);
    matrix_matrix_mult(Qhat, Uhat, V_t, ncols, rank, rank, rank);

    // V = V'
    transpose(V_t, V, rank, ncols);
  }
}
