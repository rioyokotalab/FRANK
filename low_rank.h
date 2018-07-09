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
  class LowRank : public Node {
  public:
    Dense U, S, V;
    int dim[2];
    int rank;

    LowRank();

    LowRank(const int m, const int n, const int k);

    LowRank(const LowRank &A);

    LowRank(const LowRank *A);

    LowRank(const Dense &A, const int k);

    LowRank(const NodePtr A, const int k);

    LowRank* clone() const override;

    const bool is(const int enum_id) const override;

    const char* is_string() const override;

    const Node& operator=(const double a) override;

    const Node& operator=(const Node& A) override;

    const Node& operator=(const NodePtr A) override;

    LowRank operator-() const;

    NodePtr add(const Node& B) const override;

    NodePtr sub(const Node& B) const override;

    NodePtr mul(const Node& B) const override;

    void resize(int m, int n, int k);

    const NodePtr dense() const;

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
