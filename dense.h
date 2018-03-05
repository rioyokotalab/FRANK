#ifndef dense_h
#define dense_h
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "node.h"
#include <vector>

namespace hicma {
  class LowRank;
  class Dense : public Node {
  public:
    std::vector<double> data;
    int dim[2];

    Dense();

    Dense(const int m);

    Dense(const int m, const int n);

    Dense(const Dense& A);

    double& operator[](const int i);

    const double& operator[](const int i) const;

    double& operator()(const int i, const int j);

    const double& operator()(const int i, const int j) const;

    const Dense &operator=(const Dense A);

    const Dense operator+=(const Dense& A);

    const Dense operator-=(const Dense& A);

    Dense operator+(const Dense& A) const;

    Dense operator-(const Dense& A) const;

    Dense operator*(const Dense& A) const;

    LowRank operator*(LowRank& A);

    std::vector<int> getrf() const;

    void trsm(Dense& A, const char& uplo) const;

    void gemv(Dense& A, Dense& b) const;

    void gemm(Dense& A, Dense& B) const;

    void gemm(Dense& A, LowRank& B) const;

    void gemm(LowRank& A, Dense& B) const;

    void resize(int i);

    void resize(int i, int j);

    double norm();

    void print();
  };
}
#endif
