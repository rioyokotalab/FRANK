#ifndef low_rank_h
#define low_rank_h
#include <cassert>
#include <vector>
#include "block.h"
#include "id.h"
#include "node.h"
#include "dense.h"

namespace hicma {
  class LowRank : public Node {
  public:
    Dense U, S, V;
    int dim[2];
    int rank;

    LowRank();

    LowRank(const int m, const int n, const int k);

    LowRank(const Dense& A, const int k);

    LowRank(const Block A, const int k);

    LowRank(const LowRank& A);
    LowRank(LowRank&& A);

    LowRank(const LowRank* A);

    LowRank(const Block& A);

    LowRank* clone() const override;

    friend void swap(LowRank& first, LowRank& second);

    const Node& operator=(const Node& A) override;
    const Node& operator=(Node&& A) override;
    const LowRank& operator=(LowRank A);

    const Node& operator=(Block A) override;

    const Node& operator=(const double a) override;

    LowRank operator-() const;

    Block operator+(const Node& B) const override;
    Block operator+(Block&& B) const override;
    const Node& operator+=(const Node& B) override;
    const Node& operator+=(Block&& B) override;
    Block operator-(const Node& B) const override;
    Block operator-(Block&& B) const override;
    const Node& operator-=(const Node& B) override;
    const Node& operator-=(Block&& B) override;
    Block operator*(const Node& B) const override;
    Block operator*(Block&& B) const override;

    const bool is(const int enum_id) const override;

    const char* is_string() const override;

    void resize(int m, int n, int k);

    Dense dense() const;

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
