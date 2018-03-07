#include "id.h"
#include <sys/time.h>
using namespace hicma;

int main (int argc, char** argv) {
  int ncols = 1000;
  int nrows = 1000;
  int rank = 100;
  int niter = 10;

  double *M = (double*)malloc(sizeof(double)*nrows*ncols);
  for (int i=0; i<nrows; i++) {
    for (int j=0; j<ncols; j++) {
      M[i*nrows+j] = 1 / fabs(i - j - ncols);
    }
  }

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
    form_svd_product_matrix(U,S,V_t,P, nrows, ncols, rank);
    error += get_relative_error_between_two_mats(M, P, nrows, ncols);
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
