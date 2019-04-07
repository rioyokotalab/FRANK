#ifndef hierarchical_h
#define hierarchical_h
#include "node.h"

#include <vector>

namespace hicma {

  class Dense;
  class LowRank;

  class Hierarchical : public Node {
  public:
    // NOTE: Take care to add members new members to swap
    int dim[2];
    std::vector<Any> data;

    Hierarchical();

    Hierarchical(const int m);

    Hierarchical(const int m, const int n);

    Hierarchical(const Dense& A, const int m, const int n);

    Hierarchical(const LowRank& A, const int m, const int n);

    Hierarchical(
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
                 const int rank,
                 const int nleaf,
                 const int admis=1,
                 const int ni_level=2,
                 const int nj_level=2,
                 const int i_begin=0,
                 const int j_begin=0,
                 const int i_abs=0,
                 const int j_abs=0,
                 const int level=0
                 );

    Hierarchical(const Hierarchical& A);

    Hierarchical(Hierarchical&& A);

    Hierarchical* clone() const override;

    friend void swap(Hierarchical& A, Hierarchical& B);

    const Any& operator[](const int i) const;

    Any& operator[](const int i);

    const Any& operator()(const int i, const int j) const;

    Any& operator()(const int i, const int j);

    bool is(const int enum_id) const override;

    const char* type() const override;

    double norm() const override;

    void print() const override;

    void transpose() override;

    void getrf() override;

    void trsm(const Dense& A, const char& uplo) override;

    void trsm(const Hierarchical& A, const char& uplo) override;

    void gemm(const Dense& A, const Dense& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const Dense& A, const LowRank& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const Dense& A, const Hierarchical& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const LowRank& A, const Dense& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const LowRank& A, const LowRank& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const LowRank& A, const Hierarchical& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const Hierarchical& A, const Dense& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const Hierarchical& A, const LowRank& B, const double& alpha=-1, const double& beta=1) override;

    void gemm(const Hierarchical& A, const Hierarchical& B, const double& alpha=-1, const double& beta=1) override;

    void gemm_row(
                  const Hierarchical& A, const Hierarchical& B,
                  const int& i, const int& j, const int& k_min, const int& k_max,
                  const double& alpha, const double& beta);

    void blr_col_qr(Hierarchical& Q, Dense& R);

    void split_col(Hierarchical& QL);

    void restore_col(const Hierarchical& Sp, const Hierarchical& QL);

    void col_qr(const int j, Hierarchical& Q, Hierarchical &R);

    void qr(Hierarchical& Q, Hierarchical& R);
  };
}
#endif
