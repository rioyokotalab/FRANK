#ifndef any_h
#define any_h
#include <memory>

namespace hicma {

  class Node;

  class Any {
  public:
    std::unique_ptr<Node> ptr;

    Any();

    Any(const Any& A);

    Any(const Node& A);

    Any(Node&& A);

    ~Any();

    const Any& operator=(Any A);

    bool is(const int i) const;

    const char* type() const;

    double norm() const;

    void print() const;

    void transpose();

    void getrf();

    void trsm(const Any& A, const char& uplo);

    void gemm(const Any& A, const Any& B, const double& alpha=-1, const double& beta=1);

  };
}

#endif
