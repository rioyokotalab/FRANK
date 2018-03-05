#include "id.h"
#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))
using namespace std;
namespace hicma {

  void initialize_random_matrix(double *M, int nrows, int ncols){
    time_t t;
    srand((unsigned) time(&t));
    for(int i=0; i < nrows*ncols; i++){
      M[i] = (double)rand() / (double)RAND_MAX;
    }
  }

  /* C = A^T*B 
   *
   * A - nrows_a * ncols_a
   * B - nrows_b * ncols_b
   * C - nrows_a * ncols_b
   */
  void matrix_transpose_matrix_mult(
                                    double *A,
                                    double *B,
                                    double *C,
                                    int nrows_a,
                                    int ncols_a,
                                    int nrows_b,
                                    int ncols_b)
  {
    cblas_dgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                ncols_a, ncols_b, nrows_a,
                1, A, ncols_a,
                B, ncols_b, 1,
                C, ncols_b);
  }


  /* compute compact QR factorization
   * Bt - input matrix. rank x ncols
   * Q - matrix in which Q is to be saved. ncols x rank
   * R - matrix in which R is to be saved. rank x rank
   M is mxn; Q is mxk and R is kxk
   NOTE: FUNCTION NOT USED AS OF NOW.
  */
  void compute_QR_compact_factorization(
                                        double *Bt,
                                        double *Q,
                                        double *R,
                                        int nrows,
                                        int ncols,
                                        int rank)
  {
    int k = min(ncols, rank);
    double *QR_temp = (double*)calloc(ncols*rank, sizeof(double));
    double *TAU = (double*)malloc(sizeof(double)*k);
    memcpy (QR_temp, Bt, sizeof(double)*ncols*rank);
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, ncols, rank, QR_temp, rank, TAU);
  
    for(int i=0; i<ncols; i++) {
      for(int j=0; j<rank; j++) {
        if(j>=i){
          R[i*rank + j] = QR_temp[i*rank + j];
        }
      }
    }

    for (int i = 0; i < k; ++i) Q[i*k + i] = 1.0;
        
    LAPACKE_dormqr(
                   LAPACK_ROW_MAJOR, 'L', 'N',
                   ncols, rank, rank, QR_temp, rank, TAU, Q, rank);
  }


  /* compute compact QR factoriation and get Q
   * M - original matrix. dim: nrows * rank
   * Q - Q part of QR factorized matrix. Answer is returned in this vector. nrows*k. no data during input.
   * nrows - number of rows of the Matrix Q.
   * ncols - number of columns of M.
   * rank - target rank.
   */
  void QR_factorization_getQ(double *M, double *Q, int nrows, int ncols, int rank){
    int k = min(nrows, rank);
    double *QR_temp = (double*)calloc(nrows*rank, sizeof(double)); // temp result from LAPACK stored in this
    double *TAU = (double*)calloc(k, sizeof(double)); // temp vector for LAPACK.
    memcpy (QR_temp, M, nrows*rank*sizeof(double));
    for (int i = 0;i < k; ++i) Q[i*k + i] = 1.0;
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, nrows, rank, QR_temp, rank, TAU); // correct so far
    LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'N', nrows, rank, rank, QR_temp, rank, TAU, Q, rank);
    free(TAU);
    free(QR_temp);
  }

  /* build diagonal matrix from vector elements */
  void build_diagonal_matrix(double *dvals, int n, double *D){
    for(int i=0; i<n; i++){
      D[i*n + i] = dvals[i];
    }
  }

  /* frobenius norm */
  double matrix_frobenius_norm(double *M, int nrows, int ncols){
    double norm = 0;
    for(size_t i=0; i < nrows*ncols; i++){
      double val = M[i];
      norm += val*val;
    }
  
    norm = sqrt(norm);
    return norm;
  }

  /* P = U*S*V^T */
  void form_svd_product_matrix(
                               double *U,
                               double *S,
                               double *V,
                               double *P,
                               int nrows,
                               int ncols,
                               int rank)
  {
    double * US = (double*)calloc(nrows*rank, sizeof(double));
    matrix_matrix_mult(U, S, US, nrows, rank, rank, rank);
    matrix_matrix_mult(US, V, P, nrows, rank, rank, ncols);
    free(US);
  }

  /* calculate percent error between A and B: 100*norm(A - B)/norm(A) */
  double get_relative_error_between_two_mats(double *A, double *B, int nrows, int ncols)
  {
    int i;
    double *A_minus_B = (double*)malloc(sizeof(double)*nrows*ncols);
    memcpy(A_minus_B, A, sizeof(double)*nrows*ncols);
    for (i = 0; i < nrows*ncols; ++i) {
      A_minus_B[i] = A[i] - B[i];
    }
    double normA = matrix_frobenius_norm(A, nrows, ncols);
    double normA_minus_B = matrix_frobenius_norm(A_minus_B, nrows, ncols);
    return normA_minus_B/normA;
  }

  /* C = A*B */
  void matrix_matrix_mult(double *A, double *B, double *C, int nrows_a, int ncols_a, int nrows_b, int ncols_b)
  {
    cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nrows_a, ncols_b, nrows_b,
                1, A, ncols_a, B,
                ncols_b, 1, C, ncols_b);
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
                     double *U, double *S, double *Vt,
                     double *M, int nrows, int ncols, int rank)
  {
    int lda = ncols;
    int ldu = nrows;
    int ldvt = ncols;

    double superb[min(nrows, ncols) - 1];
    LAPACKE_dgesvd(
                   LAPACK_ROW_MAJOR, 'A', 'A', rank, rank,
                   M, rank, S, U, rank, Vt, rank, superb);
  }

  void print_matrix( char* desc, int m, int n, double* a,int lda ) {
    int i, j;
    printf( "\n %s\n", desc );
    for( i = 0; i < m; i++ ) {
      for( j = 0; j < n; j++ ) printf( " %6.2f", a[i*lda + j] );
      printf( "\n" );
    }
  }

  void transpose(double * mat, double* mat_t, int nrows, int ncols)
  {
    double temp;
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        mat_t[i + j*nrows] = mat[i*ncols + j];
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
                                double *M,
                                int rank,
                                double *U,
                                double *S,
                                double *V,
                                int nrows ,
                                int ncols)
  {
    // RN = randn(n,k+p)
    // build random matrix 
    double *RN = (double*)malloc(sizeof(double)*ncols*rank);
    initialize_random_matrix(RN, ncols, rank);

    // Y = M * RN
    // multiply to get matrix of random samples Y
    double *Y = (double*)calloc(nrows*rank, sizeof(double)); // nrows * rank
    matrix_matrix_mult(M, RN, Y, nrows, ncols, ncols, rank);

    // [q, r] = qr(Y,0)
    double *Q = (double*)calloc(nrows*rank,sizeof(double));
    QR_factorization_getQ(Y, Q, nrows, ncols, rank);

    // bt = M' * q;
    // form Bt = Qt*M : rankxnrows * nrowsxncols = rankxncols
    double *Bt = (double*)calloc(ncols*rank,sizeof(double));
    matrix_transpose_matrix_mult(M, Q, Bt, nrows, ncols, nrows, rank);
    
    /* // Bt -> ncols * rank, Qhat -> ncols * rank, Rhat -> rank, rank */
    // [Qhat, Rhat] = qr(bt)
    double * Qhat = (double*)calloc(ncols*rank, sizeof(double));
    double * Rhat = (double*)calloc(rank*rank, sizeof(double));
    compute_QR_compact_factorization(Bt, Qhat, Rhat, nrows, ncols, rank);

    // compute SVD of Rhat
    // [Uhat, S, Vhat] = svd(Rhat);
    double *Uhat = (double*)calloc(rank*rank, sizeof(double));
    double *Sigmahat = (double*)calloc(rank, sizeof(double));
    double *Vhat = (double*)calloc(rank*rank, sizeof(double));
    calculate_svd(Uhat, Sigmahat, Vhat, Rhat, rank, rank, rank);

    // Vhat = Vhat'
    double *Vhat_t = (double*)calloc(rank*rank, sizeof(double));
    transpose(Vhat, Vhat_t, rank, rank);
    
    build_diagonal_matrix(Sigmahat, rank, S);

    // U = q * Vhat
    matrix_matrix_mult(Q, Vhat_t, U, nrows, rank, rank, rank);

    // V = Qhat*Uhat
    matrix_matrix_mult(Qhat, Uhat, V, ncols, rank, rank, rank);

    free(Y);
    free(Q);
    free(Uhat);
    free(Bt);
    free(Vhat);
    free(Vhat_t);
    free(Sigmahat);
  }
}
