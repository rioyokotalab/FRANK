#ifndef dense_h
#define dense_h
#include "hicma/node.h"
#include "hicma/node_proxy.h"

#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif
#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma {

  class LowRank;
  class Hierarchical;

  class Dense : public Node {
  public:
    MM_CLASS(Dense, Node);
    // NOTE: Take care to add members new members to swap
    std::vector<double> data;
    int dim[2];

    Dense();

    Dense(const int m);

    Dense(
          const int m,
          const int n,
          const int i_abs=0,
          const int j_abs=0,
          const int level=0);

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
          const int i_begin=0,
          const int j_begin=0,
          const int i_abs=0,
          const int j_abs=0,
          const int level=0);

    Dense(const Dense& A);

    Dense(Dense&& A);

    explicit Dense(const LowRank& A);

    explicit Dense(const Hierarchical& A);

    explicit Dense(const NodeProxy& A);

    Dense* clone() const override;

    Dense* move_clone() override;

    friend void swap(Dense& A, Dense& B);

    const Dense& operator=(Dense A);

    const Dense& operator=(const double a);

    Dense operator+(const Dense& A) const;

    Dense operator-(const Dense& A) const;

    const Dense& operator+=(const Dense& A);

    const Dense& operator-=(const Dense& A);

    const Dense& operator*=(const double);

    double& operator[](const int i);

    const double& operator[](const int i) const;

    double& operator()(const int i, const int j);

    const double& operator()(const int i, const int j) const;

    const char* type() const override;

    double norm() const override;

    int size() const;

    void resize(const int dim0, const int dim1);

    Dense transpose() const;

    void print() const override;

    void transpose() override;

    void svd(Dense& U, Dense& S, Dense& V);

    void svd(Dense& S);

  };

  MULTI_METHOD(make_dense, Dense, const virtual_<Node>&);

}
#endif
