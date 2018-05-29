#ifndef node_h
#define node_h
#include <iostream>
#include <memory>

namespace hicma {

  enum {
    HICMA_NODE,
    HICMA_HIERARCHICAL,
    HICMA_LOWRANK,
    HICMA_DENSE
  };

  class Node {
  public:
    const int i_abs;
    const int j_abs;
    const int level;
    Node() : i_abs(0), j_abs(0), level(0) {}
    Node(
        const int _i_abs,
        const int _j_abs,
        const int _level) : i_abs(_i_abs), j_abs(_j_abs), level(_level) {}

    virtual ~Node();

    virtual const Node& operator=(std::unique_ptr<Node> A);

    virtual const bool is(const int enum_id) const;

    virtual const char* is_string() const;

    virtual std::unique_ptr<Node> add(const Node& B) const;

    virtual std::unique_ptr<Node> sub(const Node& B) const;

    virtual std::unique_ptr<Node> mul(const Node& B) const;

    virtual void getrf_test();

    virtual void trsm(const Node& A, const char& uplo);

    virtual void gemm(const Node& A, const Node& B);
  };

}
#endif
