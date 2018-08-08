#ifndef low_rank_h
#define low_rank_h
#include <cassert>
#include <vector>
#include "block_ptr.h"
#include "id.h"
#include "node.h"
#include "dense.h"

namespace hicma {
  class _Hierarchical;
  class _LowRank : public _Node {
  public:
    Dense U, S, V;
    int dim[2];
    int rank;

    _LowRank();

    _LowRank(const int m, const int n, const int k);

    _LowRank(const _LowRank &A);

    _LowRank(const _LowRank *A);

    _LowRank(const LowRank& A);

    _LowRank(const _Dense &A, const int k);

    _LowRank(const Node A, const int k);

    _LowRank* clone() const override;

    const bool is(const int enum_id) const override;

    const char* is_string() const override;

    const _Node& operator=(const double a) override;

    const _Node& operator=(const _Node& A) override;

    const _Node& operator=(const Node& A) override;

    _LowRank operator-() const;

    Node add(const Node& B) const override;

    Node sub(const Node& B) const override;

    Node mul(const Node& B) const override;

    void resize(int m, int n, int k);

    const Node dense() const;

    double norm() const override;

    void print() const override;

    void mergeU(const _LowRank& A, const _LowRank& B);

    void mergeS(const _LowRank& A, const _LowRank& B);

    void mergeV(const _LowRank& A, const _LowRank& B);

    void trsm(const Node& A, const char& uplo) override;

    void gemm(const Node& A, const Node& B);
  };
}
#endif
