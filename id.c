#include "id.h"
#include <sys/time.h>

int main (int argc, char** argv) {
    int n = 1000;
    int m = 1000;
    int k = 16;
    int niter = 10;
    gsl_matrix *M = gsl_matrix_alloc(m,n);
    gsl_matrix *P = gsl_matrix_alloc(m,n);
    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        M->data[i*n+j] = 1 / fabs(i - j - n);
      }
    }
    gsl_matrix *U,*S,*V;
    struct timeval tic;
    gettimeofday(&tic, NULL);
    double error = 0;
    for(int it=0; it<niter; it++) {
      randomized_low_rank_svd2(M, k, &U, &S, &V);
      form_svd_product_matrix(U,S,V,P);
      error += get_percent_error_between_two_mats(M,P);
    }
    struct timeval toc;
    gettimeofday(&toc, NULL);
    double time = toc.tv_sec - tic.tv_sec + (toc.tv_usec - tic.tv_usec) * 1e-6;
    printf("time: %lf s, error: %g\n", time, error/niter);
    gsl_matrix_free(M);
    gsl_matrix_free(U);
    gsl_matrix_free(S);
    gsl_matrix_free(V);
    gsl_matrix_free(P);
    return 0;
}
