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

    LowRank(const Node& A, const int k);

    LowRank(const LowRank& A);

    LowRank(LowRank&& A);

    LowRank* clone() const override;

    friend void swap(LowRank& A, LowRank& B);

    const Node& operator=(const Node& A) override;

    const Node& operator=(Node&& A) override;

    const LowRank& operator=(LowRank A);

    const LowRank& operator+=(const LowRank& A);

    const bool is(const int enum_id) const override;

    const char* type() const override;

    double norm() const override;

    void print() const override;

    void mergeU(const LowRank& A, const LowRank& B);

    void mergeS(const LowRank& A, const LowRank& B);

    void mergeV(const LowRank& A, const LowRank& B);

    void trsm(const Node& A, const char& uplo) override;

    void gemm(const Node& A, const Node& B);
  };
}
#endif
