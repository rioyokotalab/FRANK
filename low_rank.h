#ifndef low_rank_h
#define low_rank_h
#include "dense.h"

namespace hicma {

  class LowRank : public Node {
  public:
    // NOTE: Take care to add members new members to swap
    Dense U, S, V;
    int dim[2];
    int rank;

    LowRank();

    LowRank(
            const int m,
            const int n,
            const int k,
            const int i_abs=0,
            const int j_abs=0,
            const int level=0);

    LowRank(const Dense& A, const int k);

    LowRank(const LowRank& A);

    LowRank(LowRank&& A);

    LowRank(const Any& _A, const int k=1);

    LowRank* clone() const override;

    friend void swap(LowRank& A, LowRank& B);

    const LowRank& operator=(LowRank A);

    const LowRank& operator+=(const LowRank& A);

    bool is(const int enum_id) const override;

    const char* type() const override;

    double norm() const override;

    void print() const override;

    void mergeU(const LowRank& A, const LowRank& B);

    void mergeS(const LowRank& A, const LowRank& B);

    void mergeV(const LowRank& A, const LowRank& B);

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
  };
}
#endif
