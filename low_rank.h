#ifndef low_rank_h
#define low_rank_h
#include <cassert>
#include "id.h"
#include "node.h"
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

    LowRank(const LowRank &A);

    LowRank(const Dense &D, const int k);

    const LowRank& operator=(const LowRank A);

    Dense operator+=(const Dense& D) const;

    Dense operator-=(const Dense& D) const;

    Dense operator+(const Dense& D) const;

    Dense operator-(const Dense& D) const;

    LowRank operator+=(const LowRank& A) const;

    LowRank operator-=(const LowRank& A) const;

    LowRank operator+(const LowRank& A) const;

    LowRank operator-(const LowRank& A) const;

    LowRank operator*(const Dense& D);

    LowRank operator*(const LowRank& A);

    LowRank operator-() const;

    void resize(int m, int n, int k);

    void trsm(Dense& A, const char& uplo);

    LowRank& gemm(const Dense& A, const Dense& B);

    LowRank& gemm(const Dense& A, const LowRank& B);

    LowRank& gemm(const LowRank& A, const Dense& B);

    LowRank& gemm(const LowRank& A, const LowRank& B);

    void mergeU(const LowRank&A, const LowRank& B);

    void mergeS(const LowRank&A, const LowRank& B);

    void mergeV(const LowRank&A, const LowRank& B);

    Dense dense() const;
  };
}
#endif
