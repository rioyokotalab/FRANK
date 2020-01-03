#ifndef hicma_classes_dense_h
#define hicma_classes_dense_h

#include "hicma/classes/node.h"

#include <vector>
#include <memory>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma {

  class NodeProxy;
  class LowRank;
  class Hierarchical;

  class Dense : public Node {
  private:
    std::vector<double> data;
  public:
    MM_CLASS(Dense, Node);
    int dim[2];

    // Special member functions
    Dense();

    ~Dense();

    Dense(const Dense& A);

    Dense& operator=(const Dense& A);

    Dense(Dense&& A);

    Dense& operator=(Dense&& A);

    // Overridden functions from Node
    std::unique_ptr<Node> clone() const override;

    std::unique_ptr<Node> move_clone() override;

    const char* type() const override;

    // Explicit conversions using multiple-dispatch function.
    explicit Dense(const Node& A);

    // Additional constructors
    Dense(int m);

    Dense(
      int m, int n,
      int i_abs=0, int j_abs=0,
      int level=0
    );

    Dense(
      void (*func)(
        std::vector<double>& data,
        std::vector<double>& x,
        int ni, int nj,
        int i_begin, int j_begin
      ),
      std::vector<double>& x,
      int ni, int nj,
      int i_begin=0, int j_begin=0,
      int i_abs=0, int j_abs=0,
      int level=0
    );

    Dense(
      void (*func)(
        std::vector<double>& data,
        std::vector<std::vector<double>>& x,
        const int& ni, const int& nj,
        const int& i_begin, const int& j_begin
      ),
      std::vector<std::vector<double>>& x,
      const int ni, const int nj,
      const int i_begin=0, const int j_begin=0,
      const int i_abs=0, const int j_abs=0,
      const int level=0
    );

    // Additional operators
    const Dense& operator=(const double a);

    Dense operator+(const Dense& A) const;

    Dense operator-(const Dense& A) const;

    const Dense& operator+=(const Dense& A);

    const Dense& operator-=(const Dense& A);

    const Dense& operator*=(const double);

    double& operator[](int i);

    const double& operator[](int i) const;

    double& operator()(int i, int j);

    const double& operator()(int i, int j) const;

    // Utility methods
    int size() const;

    void resize(int dim0, int dim1);

    Dense transpose() const;

    void transpose();

  };

  MULTI_METHOD(make_dense, Dense, const virtual_<Node>&);

} // namespace hicma

#endif // hicma_classes_dense_h
