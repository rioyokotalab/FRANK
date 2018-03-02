#include "id.h"
#include <sys/time.h>

int main (int argc, char** argv) {
    int ncols = 1000;
    int nrows = 1000;
    int k = 16;
    int niter = 10;
    double *M = (double*)malloc(nrows*ncols);
    double *P = (double*)malloc(nrows*ncols);
    for (int i=0; i<nrows; i++) {
      for (int j=0; j<ncols; j++) {
        M->data[i*n+j] = 1 / fabs(i - j - n);
      }
    }
    double *U,*S,*V;
    struct timeval tic;
    gettimeofday(&tic, NULL);
    double error = 0;
    for(int it=0; it<niter; it++) {
      randomized_low_rank_svd2(M, k, &U, &S, &V, nrows, ncols);
      form_svd_product_matrix(U,S,V,P);
      error += get_percent_error_between_two_mats(M,P);
    }
    struct timeval toc;
    gettimeofday(&toc, NULL);
    double time = toc.tv_sec - tic.tv_sec + (toc.tv_usec - tic.tv_usec) * 1e-6;
    printf("time: %lf s, error: %g\n", time, error/niter);
    free(M);
    free(U);
    free(S);
    free(V);
    free(P);
    return 0;
}
