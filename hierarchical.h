#ifndef hierarchical_h
#define hierarchical_h
#include "low_rank.h"
#include "print.h"

#include <algorithm>

namespace hicma {

  class Hierarchical : public Node {
  public:
    // NOTE: Take care to add members new members to swap
    int dim[2];
    std::vector<Block> data;

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

    Hierarchical(const Hierarchical* A);

    Hierarchical* clone() const override;

    friend void swap(Hierarchical& A, Hierarchical& B);

    Block& operator[](const int i);

    const Node& operator()(const int i, const int j) const;

    Block& operator()(const int i, const int j);

    const bool is(const int enum_id) const override;

    const char* type() const override;

    double norm() const override;

    void print() const override;

    void getrf() override;

    void trsm(const Node& A, const char& uplo) override;

    void gemm(const Node& A, const Node& B) override;

    void gemm_row(
        const Hierarchical& A,
        const Hierarchical& B,
        const int i, const int j, const int k_min, const int k_max);
  };
}
#endif
