/* single core code using GSL library */
#include "low_rank_svd_algorithms_gsl.h"
#include <sys/time.h>

int main (int argc, char** argv) {
    int m, n, k;
    double normM, normU, normS, normV, normP;
    gsl_matrix *U,*S,*V;
    k = 10;
    m = 1000; n = 1000;
    gsl_matrix *M = gsl_matrix_alloc(m,n);
    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        M->data[i*n+j] = 1 / fabs(i - j - n);
      }
    }
    struct timeval tic;
    gettimeofday(&tic, NULL);
    for(int it=0; it<1; it++) {
      switch(atoi(argv[1])) {
      case 0 :
        randomized_low_rank_svd1(M, k, &U, &S, &V);
        break;
      case 1 :
        randomized_low_rank_svd2(M, k, &U, &S, &V);
        break;
      case 2 :
        randomized_low_rank_svd3(M, k, 3, 1, &U, &S, &V);
        break;
      case 3 :
        randomized_low_rank_svd2_autorank1(M, 0.5, 0.0001, &U, &S, &V);
        break;
      case 4 :
        randomized_low_rank_svd2_autorank2(M, k, 0.1, &U, &S, &V);
        break;
      case 5 :
        randomized_low_rank_svd3_autorank2(M, k, 0.1, 3, 1, &U, &S, &V);
      }
    }
    struct timeval toc;
    gettimeofday(&toc, NULL);
    gsl_matrix *P = gsl_matrix_alloc(m,n);
    form_svd_product_matrix(U,S,V,P);
    double time = toc.tv_sec - tic.tv_sec + (toc.tv_usec - tic.tv_usec) * 1e-6;
    double error = get_percent_error_between_two_mats(M,P);
    printf("time: %lf s, error: %lf\n", time, error);

    gsl_matrix_free(M);
    gsl_matrix_free(U);
    gsl_matrix_free(S);
    gsl_matrix_free(V);
    gsl_matrix_free(P);

    return 0;
}
