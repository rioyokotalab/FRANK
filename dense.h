#ifndef dense_h
#define dense_h
#include "block.h"
#include "node.h"

#include <vector>

namespace hicma {
  class Dense : public Node {
  public:
    std::vector<double> data;
    int dim[2];

    Dense();

    Dense(const int m);

    Dense(const int m, const int n);

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

    Dense(const Dense& A);
    Dense(Dense&& A);

    Dense(const Dense* A);

    Dense(const Block& A);

    Dense* clone() const override;

    // NOTE: Take care to add members new members to swap
    friend void swap(Dense& first, Dense& second);

    const Node& operator=(const Node& A) override;
    const Node& operator=(Node&& A) override;
    const Dense& operator=(Dense A);

    const Node& operator=(Block A) override;

    const Node& operator=(const double a) override;

    Dense operator-() const;

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

    double& operator[](const int i);

    const double& operator[](const int i) const;

    double& operator()(const int i, const int j);

    const double& operator()(const int i, const int j) const;

    const bool is(const int enum_id) const override;

    const char* is_string() const override;

    void resize(int i);

    void resize(int i, int j);

    Dense extract(int i, int j, int ni, int nj);

    double norm() const override;

    void print() const override;

    void getrf() override;

    void trsm(const Node& A, const char& uplo) override;

    void gemm(const Node& A, const Node& B);
  };
}
#endif
