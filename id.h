#ifndef id_h
#define id_h

/* NOTE TO THE WISE
 *
 * You MUST tranpose V that you get from the ID function after the computation is done.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <lapacke.h>
#include <cblas.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

namespace hicma {
  void matrix_matrix_mult(
                          double *A, double *B, double *C,
                          int nrows_a, int ncols_a, int nrows_b, int ncols_b);

  void print_matrix( char* desc, int m, int n, double* a,int lda );

  void initialize_random_matrix(double *M, int nrows, int ncols);

  void compute_QR_compact_factorization(
                                        double *Bt,
                                        double *Q,
                                        double *R,
                                        int nrows,
                                        int ncols,
                                        int rank);

  void QR_factorization_getQ(double *M, double *Q, int nrows, int ncols, int rank);

  void build_diagonal_matrix(double *dvals, int n, double *D);

  double matrix_frobenius_norm(double *M, int nrows, int ncols);

  void form_svd_product_matrix(
                               double *U,
                               double *S,
                               double *V,
                               double *P,
                               int nrows,
                               int ncols,
                               int rank);

  double get_relative_error_between_two_mats(double *A, double *B, int nrows, int ncols);

  void calculate_svd(
                     double *U, double *S, double *Vt,
                     double *M, int nrows, int ncols, int rank);

  void randomized_low_rank_svd2(
                                double *M,
                                int rank,
                                double *U,
                                double *S,
                                double *V,
                                int nrows ,
                                int ncols);

  void transpose(double * mat, double* mat_t, int nrows, int ncols);
}
#endif
