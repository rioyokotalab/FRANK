#include "id.h"
#include <sys/time.h>

int main (int argc, char** argv) {
  // int ncols = 1000;
  // int nrows = 1000;
  // int k = 16;
  int niter = 10;
  //double *M = (double*)malloc(sizeof(double)*nrows*ncols);
  //double *P = (double*)malloc(sizeof(double)*nrows*ncols);
  // for (int i=0; i<nrows; i++) {
  //   for (int j=0; j<ncols; j++) {
  //     M[i*nrows+j] = 1 / fabs(i - j - ncols);
  //   }
  // }
  double M[30] = {
    8.79,   9.93,   9.83,   5.45,   3.16,
    6.11,   6.91,   5.04,  -0.27,   7.98,
    -9.15,  -7.93,   4.86,   4.85 ,  3.01,
    9.57,  1.64,   8.83,   0.74,   5.80,
    -3.49,   4.02,   9.80,  10.00,   4.27,
    9.84,   0.15,  -8.99,  -6.02,  -5.31 };
  int nrows = 6;
  int ncols = 5;
  int rank = 3;
  double *P = (double*)malloc(sizeof(double)*nrows*ncols);
  
  double *U,*S,*V;
  struct timeval tic;
  gettimeofday(&tic, NULL);
  double error = 0;
  for(int it=0; it<niter; it++) {
    randomized_low_rank_svd2(M, rank, U, S, V, nrows, ncols);
    form_svd_product_matrix(U,S,V,P, nrows, ncols, rank);
    error += get_percent_error_between_two_mats(M, P, nrows, ncols);
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
