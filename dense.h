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
  class Hierarchical;
  class Dense : public Node {
  public:
    std::vector<double> data;
    int dim[2];

    Dense();

    Dense(const int m);

    Dense(const int m, const int n);

    Dense(const Dense& A);

    Dense(
        const Hierarchical* parent,
        const int i_rel,
        const int j_rel,
        const size_t xi_half,
        const size_t xj_half);

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

    void trsm(Dense& A, const char& uplo);

    void gemv(const Dense& A, const Dense& b) const;

    void gemv(const LowRank& A, const Dense& b) const;

    void gemm(const Dense& A, const Dense& B) const;

    void gemm(const Dense& A, const LowRank& B) const;

    void gemm(const LowRank& A, const Dense& B) const;

    void gemm(const LowRank& A, const LowRank& B) const;

    void resize(int i);

    void resize(int i, int j);

    double norm();

    void print();
  };
}
#endif
