#ifndef hierarchical_h
#define hierarchical_h
#include <boost/any.hpp>
#include <memory>

namespace hicma {
  class Node;
  class Dense;
  class LowRank;
  class Hierarchical : public Node {
  public:
    int dim[2];
    std::vector<boost::any> data;
    std::vector<std::unique_ptr<Node>> data_test;

    Hierarchical();

    Hierarchical(const int m);

    Hierarchical(const int m, const int n);

    Hierarchical(const Hierarchical& A);

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

    virtual const bool is(const int enum_id) const override;

    virtual const char* is_string() const override;

    boost::any& operator[](const int i);

    const boost::any& operator[](const int i) const;

    boost::any& operator()(const int i, const int j);

    const boost::any& operator()(const int i, const int j) const;

    const Hierarchical& operator=(const double a);

    const Hierarchical& operator=(const Hierarchical& A);

    const Dense operator+=(const Dense& A);

    const LowRank operator+=(const LowRank& A);

    const Hierarchical operator+=(const Hierarchical& A);

    const Dense operator-=(const Dense& A);

    const LowRank operator-=(const LowRank& A);

    const Hierarchical operator-=(const Hierarchical& A);

    const Dense operator*=(const Dense& A);

    const LowRank operator*=(const LowRank& A);

    const Hierarchical operator*=(const Hierarchical& A);

    Dense operator+(const Dense& A) const;

    LowRank operator+(const LowRank& A) const;

    Hierarchical operator+(const Hierarchical& A) const;

    Dense operator-(const Dense& A) const;

    LowRank operator-(const LowRank& A) const;

    Hierarchical operator-(const Hierarchical& A) const;

    Dense operator*(const Dense& A) const;

    LowRank operator*(const LowRank& A) const;

    Hierarchical operator*(const Hierarchical& A) const;

    Dense dense() const;

    Dense lowRank() const;

    Dense& dense(const int i);

    Dense& dense(const int i, const int j);

    double norm();

    void print() const;

    void getrf();

    void trsm(const Hierarchical& A, const char& uplo);

    void gemm(const Hierarchical& A, const Hierarchical& B);
  };
}
#endif
