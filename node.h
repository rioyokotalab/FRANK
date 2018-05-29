#ifndef node_h
#define node_h
#include <iostream>

namespace hicma {
  enum {
    HICMA_NODE,
    HICMA_HIERARCHICAL,
    HICMA_DENSE,
    HICMA_LOWRANK
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

    virtual ~Node() {};

    virtual const bool is(const int enum_id) const {
      return enum_id == HICMA_NODE;
    }

    virtual const char* is_string() const { return "Node"; }

    virtual void getrf_test() {};

    virtual void trsm_test(const Node& A, const char& uplo) {};

    virtual void gemm_test(const Node& A, const Node& B) {};
  };
}
#endif
