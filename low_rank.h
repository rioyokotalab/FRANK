#ifndef low_rank_h
#define low_rank_h
#include <cassert>
#include <vector>
#include "id.h"
#include "node.h"
#include "dense.h"
#include "block_ptr.h"

namespace hicma {
  class Hierarchical;
  class LowRank : public _Node {
  public:
    DensePtr U, S, V;
    int dim[2];
    int rank;

    LowRank();

    LowRank(const int m, const int n, const int k);

    LowRank(const LowRank &A);

    LowRank(const LowRank *A);

    LowRank(const LowRankPtr& A);

    LowRank(const Dense &A, const int k);

    LowRank(const Node A, const int k);

    LowRank* clone() const override;

    const bool is(const int enum_id) const override;

    const char* is_string() const override;

    const _Node& operator=(const double a) override;

    const _Node& operator=(const _Node& A) override;

    const _Node& operator=(const Node& A) override;

    LowRank operator-() const;

    Node add(const Node& B) const override;

    Node sub(const Node& B) const override;

    Node mul(const Node& B) const override;

    void resize(int m, int n, int k);

    const Node dense() const;

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
