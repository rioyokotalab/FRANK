#ifndef id_h
#define id_h

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>

namespace hicma {
  void initialize_random_matrix(gsl_matrix *M);

  void matrix_matrix_mult(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C);

  void matrix_transpose_matrix_mult(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C);

  void compute_QR_compact_factorization(gsl_matrix *M, gsl_matrix *Q, gsl_matrix *R);

  void QR_factorization_getQ(gsl_matrix *M, gsl_matrix *Q);

  void build_diagonal_matrix(gsl_vector *dvals, int n, gsl_matrix *D);

  double matrix_frobenius_norm(gsl_matrix *M);

  void form_svd_product_matrix(gsl_matrix *U, gsl_matrix *S, gsl_matrix *V, gsl_matrix *P);

  double get_percent_error_between_two_mats(gsl_matrix *A, gsl_matrix *B);

  void randomized_low_rank_svd2(gsl_matrix *M, int k, gsl_matrix **U, gsl_matrix **S, gsl_matrix **V);
}
#endif
