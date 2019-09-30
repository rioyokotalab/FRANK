#ifndef dense_h
#define dense_h
#include "node.h"
#include "any.h"

#include <vector>
#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

namespace hicma {

  class LowRank;
  class Hierarchical;

  class Dense : public Node {
  public:
    // NOTE: Take care to add members new members to swap
    std::vector<double> data;
    int dim[2];

    Dense();

    Dense(const int m);

    Dense(
          const int m,
          const int n,
          const int i_abs=0,
          const int j_abs=0,
          const int level=0);

    Dense(
          void (*func)(
                       std::vector<double>& data,
                       std::vector<double>& x,
                       const int& ni,
                       const int& nj,
                       const int& i_begin,
                       const int& j_begin
                       ),
          std::vector<double>& x,
          const int ni,
          const int nj,
          const int i_begin=0,
          const int j_begin=0,
          const int i_abs=0,
          const int j_abs=0,
          const int level=0);

    Dense(const Dense& A);

    Dense(Dense&& A);

    Dense(const LowRank& A);

    Dense(const Hierarchical& A);

    Dense(const Any& A);

    Dense* clone() const override;

    friend void swap(Dense& A, Dense& B);

    const Dense& operator=(Dense A);

    const Dense& operator=(const double a);

    Dense operator+(const Dense& A) const;

    Dense operator-(const Dense& A) const;

    const Dense& operator+=(const Dense& A);

    const Dense& operator-=(const Dense& A);

    double& operator[](const int i);

    const double& operator[](const int i) const;

    double& operator()(const int i, const int j);

    const double& operator()(const int i, const int j) const;

    bool is(const int enum_id) const override;

    const char* type() const override;

    double norm() const override;

    int size() const;

    void resize(const int dim0, const int dim1);

    Dense transpose() const;

    void print() const override;

    void transpose() override;

    void getrf() override;

    void trsm(const Dense& A, const char& uplo) override;

    void trsm(const Hierarchical& A, const char& uplo) override;

    void gemm(const Dense& A, const Dense&B, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
              const double& alpha, const double& beta);

    void gemm(const Dense& A, const Dense& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const Dense& A, const LowRank& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const Dense& A, const Hierarchical& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const LowRank& A, const Dense& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const LowRank& A, const LowRank& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const LowRank& A, const Hierarchical& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const Hierarchical& A, const Dense& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const Hierarchical& A, const LowRank& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const Hierarchical& A, const Hierarchical& B, const double& alpha=-1, const double& beta=1) override;

    void qr(Dense& Q, Dense& R);

    void svd(Dense& U, Dense& S, Dense& V);

    void svd(Dense& S);

    void geqrt(Dense& T) override;

    void geqrt2(Dense& T);

    void larfb(const Dense& Y, const Dense& T, const bool trans=false) override;

    void larfb(const Hierarchical& Y, const Hierarchical& T, const bool trans=false) override;

    void tpqrt(Dense& A, Dense& T) override;

    void tpqrt(Hierarchical& A, Dense& T) override;

    void tpqrt(Hierarchical& A, Hierarchical& T) override;

    void tpmqrt(Dense& B, const Dense& Y, const Dense& T, const bool trans=false) override;

    void tpmqrt(Dense& B, const LowRank& Y, const Dense& T, const bool trans=false) override;

    void tpmqrt(LowRank& B, const Dense& Y, const Dense& T, const bool trans=false) override;

    void tpmqrt(LowRank& B, const LowRank& Y, const Dense &T, const bool trans=false) override;

    void tpmqrt(Hierarchical& B, const Dense& Y, const Dense& T, const bool trans=false) override;

    void tpmqrt(Hierarchical& B, const Hierarchical& Y, const Hierarchical& T, const bool trans=false) override;

  };
}
#endif