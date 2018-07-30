#ifndef dense_h
#define dense_h
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "node.h"
#include "block_ptr.h"

namespace hicma {
  class Dense : public Node {
  public:
    std::vector<double> data;
    int dim[2];

    Dense();

    Dense(const int m);

    Dense(const int m, const int n);

    Dense(const Dense& A);

    Dense(const Dense* A);

    Dense(const DensePtr& A);

    Dense(
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
          const int i_begin,
          const int j_begin,
          const int _i_abs=0,
          const int _j_abs=0,
          const int level=0
          );

    Dense* clone() const override;

    const bool is(const int enum_id) const override;

    const char* is_string() const override;

    double& operator[](const int i);

    const double& operator[](const int i) const;

    double& operator()(const int i, const int j);

    const double& operator()(const int i, const int j) const;

    const Node& operator=(const double a) override;

    const Node& operator=(const Node& A) override;

    const Node& operator=(const NodePtr& A) override;

    Dense operator-() const;

    NodePtr add(const NodePtr& B) const override;

    NodePtr sub(const NodePtr& B) const override;

    NodePtr mul(const NodePtr& B) const override;

    void resize(int i);

    void resize(int i, int j);

    Dense extract(int i, int j, int ni, int nj);

    double norm() const override;

    void print() const override;

    void getrf() override;

    void trsm(const NodePtr& A, const char& uplo) override;

    void gemm(const NodePtr& A, const NodePtr& B);
  };
}
#endif
