#ifndef low_rank_h
#define low_rank_h
#include <cassert>
#include "id.h"
#include <vector>

namespace hicma {
  class Node;
  class Dense;
  class Hierarchical;
  class LowRank : public Node {
  public:
    Dense U, S, V;
    int dim[2];
    int rank;

    LowRank();

    LowRank(const int m, const int n, const int k);

    LowRank(const LowRank &A);

    LowRank(const Dense &A, const int k);

    LowRank* clone() const override;

    const bool is(const int enum_id) const override;

    const char* is_string() const override;

    const LowRank& operator=(const double v);

    const Node& assign(const double a) override;

    const LowRank& operator=(const LowRank A);

    const Node& operator=(const Node& A) override;

    const Node& operator=(const std::shared_ptr<Node> A) override;

    const Dense operator+=(const Dense& D);

    const LowRank operator+=(const LowRank& A);

    const LowRank operator+=(const Hierarchical& A);

    const Dense operator-=(const Dense& D);

    const LowRank operator-=(const LowRank& A);

    const LowRank operator-=(const Hierarchical& A);

    const LowRank operator*=(const Dense& D);

    const LowRank operator*=(const LowRank& A);

    const LowRank operator*=(const Hierarchical& A);

    Dense operator+(const Dense& D) const;

    LowRank operator+(const LowRank& A) const;

    LowRank operator+(const Hierarchical& A) const;

    Dense operator-(const Dense& D) const;

    LowRank operator-(const LowRank& A) const;

    LowRank operator-(const Hierarchical& A) const;

    LowRank operator*(const Dense& D) const;

    LowRank operator*(const LowRank& A) const;

    LowRank operator*(const Hierarchical& A) const;

    LowRank operator-() const;

    std::shared_ptr<Node> add(const Node& B) const override;

    std::shared_ptr<Node> sub(const Node& B) const override;

    std::shared_ptr<Node> mul(const Node& B) const override;

    void resize(int m, int n, int k);

    Dense dense() const;

    double norm() const;

    double norm_test() const override;

    void print() const;

    void mergeU(const LowRank& A, const LowRank& B);

    void mergeS(const LowRank& A, const LowRank& B);

    void mergeV(const LowRank& A, const LowRank& B);

    void trsm(const Dense& A, const char& uplo);

    void trsm(const Node& A, const char& uplo) override;

    void gemm(const Dense& A, const LowRank& B);

    void gemm(const LowRank& A, const Dense& B);

    void gemm(const LowRank& A, const LowRank& B);

    void gemm(const Node& A, const Node& B);
  };
}
#endif
