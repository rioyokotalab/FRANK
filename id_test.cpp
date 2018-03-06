#include "id.h"
#include <sys/time.h>
using namespace hicma;

void transpose(double * mat, double* mat_t, int nrows, int ncols)
{
  int index = 0;
  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      mat_t[index] = mat[i + j*nrows];
      index += 1;
    }
  }
}

int main (int argc, char** argv) {
  // int ncols = 100;
  // int nrows = 100;
  // int rank = 50;
  // int niter = 1;
  // double *M = (double*)malloc(sizeof(double)*nrows*ncols);
  // for (int i=0; i<nrows; i++) {
  //   for (int j=0; j<ncols; j++) {
  //     M[i*nrows+j] = 1 / fabs(i - j - ncols);
  //   }
  // }

  //  printf("m: M[0] %f, M[1] %f\n", M[0], M[1]);

  double M[30] =  {8.79,  9.93,  9.83, 5.45,  3.16,
               6.11,  6.91,  5.04, -0.27, 7.98,
               -9.15, -7.93,  4.86, 4.85,  3.01,
               9.57,  1.64,  8.83, 0.74,  5.80,
               -3.49,  4.02,  9.80, 10.00,  4.27,
               9.84,  0.15, -8.99, -6.02, -5.31};
  int nrows = 6;
  int ncols = 5;
  int rank = 4;
  int niter = 1;

  struct timeval tic;
  gettimeofday(&tic, NULL);
  double error = 0;
  for(int it=0; it<niter; it++) {
    double * P = (double*)calloc(nrows*ncols, sizeof(double));
    double * U = (double*)calloc(nrows*rank,sizeof(double));
    double * S = (double*)calloc(rank*rank,sizeof(double));
    double * V = (double*)calloc(rank*ncols,sizeof(double));
    double * V_t = (double*)calloc(ncols*rank,sizeof(double));
  
    randomized_low_rank_svd2(M, rank, U, S, V, nrows, ncols);
    transpose(V,V_t,rank, ncols);
    printf("SVD product:\n");
    print_matrix("U", nrows, rank, U, rank);
    print_matrix("S", rank, rank, S, rank);
    print_matrix("V", ncols, rank, V, rank);
    print_matrix("V_t", rank, ncols, V_t, ncols);
    
    form_svd_product_matrix(U,S,V_t,P, nrows, ncols, rank);
    
    error = get_relative_error_between_two_mats(M, P, nrows, ncols);
    print_matrix("P", nrows, ncols, P, ncols);

    free(P);
    free(U);
    free(S);
    free(V);
  }
  struct timeval toc;
  gettimeofday(&toc, NULL);
  double time = toc.tv_sec - tic.tv_sec + (toc.tv_usec - tic.tv_usec) * 1e-6;
  printf("time: %lf s, error: %g\n", time, error/niter);
  return 0;
}
