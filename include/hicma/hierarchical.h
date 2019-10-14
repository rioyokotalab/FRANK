#ifndef hierarchical_h
#define hierarchical_h
#include "hicma/node.h"
#include "hicma/node_proxy.h"

#include <vector>

#include "yorel/multi_methods.hpp"

namespace hicma {

  class Dense;
  class LowRank;

  class Hierarchical : public Node {
  public:
    MM_CLASS(Hierarchical, Node);
    // NOTE: Take care to add members new members to swap
    int dim[2];
    std::vector<NodeProxy> data;

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

    const NodeProxy& operator[](const int i) const;

    NodeProxy& operator[](const int i);

    const NodeProxy& operator()(const int i, const int j) const;

    NodeProxy& operator()(const int i, const int j);

    const char* type() const override;

    double norm() const override;

    void print() const override;

    void transpose() override;

    // void blr_col_qr(Hierarchical& Q, Hierarchical& R);

    // void split_col(Hierarchical& QL);

    // void restore_col(const Hierarchical& Sp, const Hierarchical& QL);

    // void col_qr(const int j, Hierarchical& Q, Hierarchical &R);

    // void qr(Hierarchical& Q, Hierarchical& R);

    // void geqrt(Hierarchical& T) override;

    // void larfb(const Dense& Y, const Dense& T, const bool trans=false) override;

    // void larfb(const Hierarchical& Y, const Hierarchical& T, const bool trans=false) override;

    // void tpqrt(Dense& A, Dense& T) override;

    // void tpqrt(Hierarchical& A, Hierarchical& T) override;

  };
}
#endif
