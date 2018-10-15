#ifndef functions_h
#define functions_h
#include <cmath>
#include <cstdlib>
#include <vector>
#include <mkl.h>

extern "C" lapack_int LAPACKE_latms(int matrix_layout, lapack_int m, lapack_int n, char dist, lapack_int *iseed, char sym, double *d, lapack_int mode, double cond, double dmax, lapack_int kl, lapack_int ku, char pack, double *a, lapack_int lda)
{ return LAPACKE_dlatms(matrix_layout, m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack, a, lda); }

namespace hicma {
  void zeros(
            std::vector<double>& data,
            std::vector<double>& x,
            const int& ni,
            const int& nj,
            const int& i_begin,
            const int& j_begin
            ) {
    for (int i=0; i<ni; i++) {
      for (int j=0; j<nj; j++) {
        data[i*nj+j] = 0;
      }
    }
  }

  void random(
              std::vector<double>& data,
              std::vector<double>& x,
              const int& ni,
              const int& nj,
              const int& i_begin,
              const int& j_begin
              ) {
    for (int i=0; i<ni; i++) {
      for (int j=0; j<nj; j++) {
        data[i*nj+j] = drand48();
      }
    }
  }

  void latms(
             std::vector<double>& data,
             std::vector<double>& x,
             const int& ni,
             const int& nj,
             const int& i_begin,
             const int& j_begin
             ) {
    int iseed[4];
    for (int i=0; i<4; i++) iseed[i] = rand() % 4096;
    if(iseed[3] % 2 != 1) iseed[3]++;
    double s[ni];
    for(int i=0; i<ni; i++)
      s[i] = exp(-x[0] * i);
    LAPACKE_latms(LAPACK_ROW_MAJOR, ni, nj, 'N', iseed, 'N', s, 0, 0, 1, ni, nj, 'N', &data[0], nj);
  }

  void arange(
            std::vector<double>& data,
            std::vector<double>& x,
            const int& ni,
            const int& nj,
            const int& i_begin,
            const int& j_begin
            ) {
    for (int i=0; i<ni; i++) {
      for (int j=0; j<nj; j++) {
        data[i*nj+j] = (double)(i*nj+j);
      }
    }
  }

  void laplace1d(
                 std::vector<double>& data,
                 std::vector<double>& x,
                 const int& ni,
                 const int& nj,
                 const int& i_begin,
                 const int& j_begin
                 ) {
    for (int i=0; i<ni; i++) {
      for (int j=0; j<nj; j++) {
        data[i*nj+j] = 1 / (std::abs(x[i+i_begin] - x[j+j_begin]) + 1e-3);
      }
    }
  }
}
#endif
