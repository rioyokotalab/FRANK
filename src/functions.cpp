#ifndef functions_h
#define functions_h

#include "hicma/functions.h"

#include <cmath>
#include <vector>

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

  void identity(
                std::vector<double>& data,
                std::vector<double>& x,
                const int& ni,
                const int& nj,
                const int& i_begin,
                const int& j_begin
                ) {
    for (int i=0; i<ni; i++) {
      for (int j=0; j<nj; j++) {
        data[i*nj+j] = i_begin+i == j_begin+j ? 1 : 0;
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
