#ifndef dense_h
#define dense_h
#include "node.h"

#include <iomanip>
#include <vector>

namespace hicma {

  class Dense : public Node {
  public:
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
          const int i_begin,
          const int j_begin,
          const int i_abs=0,
          const int j_abs=0,
          const int level=0);

    Dense(const Dense& A);

    Dense(Dense&& A);

    Dense(const Dense* A);

    Dense(const Block& A);

    Dense* clone() const override;

    friend void swap(Dense& A, Dense& B);

    const Dense& operator=(Dense A);

    const Dense& operator=(const double a);

    Dense operator+(const Dense& A) const;

    const Dense& operator+=(const Dense& A);

    double& operator[](const int i);

    const double& operator[](const int i) const;

    double& operator()(const int i, const int j);

    const double& operator()(const int i, const int j) const;

    const bool is(const int enum_id) const override;

    const char* type() const override;

    double norm() const override;

    void print() const override;

    void getrf() override;

    void trsm(const Node& A, const char& uplo) override;

    void gemm(const Node& A, const Node& B) override;
  };
}
#endif
