#ifndef node_h
#define node_h

namespace hicma {

  enum {
    HICMA_NODE,
    HICMA_HIERARCHICAL,
    HICMA_LOWRANK,
    HICMA_DENSE
  };

  class Dense;
  class LowRank;
  class Hierarchical;

  class Node {

  public:
    int i_abs; //! Row number of the node on the current recursion level
    int j_abs; //! Column number of the node on the current recursion level
    int level; //! Recursion level of the node

    Node();

    Node(const int _i_abs, const int _j_abs, const int _level);

    Node(const Node& A);

    virtual ~Node();

    virtual Node* clone() const;

    const Node& operator=(Node&& A);

    virtual bool is(const int enum_id) const;

    virtual const char* type() const;

    virtual double norm() const;

    virtual void print() const;

    virtual void getrf();

    virtual void trsm(const Dense& A, const char& uplo);

    virtual void trsm(const Hierarchical& A, const char& uplo);

    virtual void gemm(const Dense& A, const Dense& B, const double& alpha=-1, const double& beta=1);

    virtual void gemm(const Dense& A, const LowRank& B, const double& alpha=-1, const double& beta=1);

    virtual void gemm(const Dense& A, const Hierarchical& B, const double& alpha=-1, const double& beta=1);

    virtual void gemm(const LowRank& A, const Dense& B, const double& alpha=-1, const double& beta=1);

    virtual void gemm(const LowRank& A, const LowRank& B, const double& alpha=-1, const double& beta=1);

    virtual void gemm(const LowRank& A, const Hierarchical& B, const double& alpha=-1, const double& beta=1);

    virtual void gemm(const Hierarchical& A, const Dense& B, const double& alpha=-1, const double& beta=1);

    virtual void gemm(const Hierarchical& A, const LowRank& B, const double& alpha=-1, const double& beta=1);

    virtual void gemm(const Hierarchical& A, const Hierarchical& B, const double& alpha=-1, const double& beta=1);
  };
}
#endif
