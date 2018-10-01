#ifndef node_h
#define node_h
#include "mpi_utils.h"
#include <iostream>
#include <map>
#include <tuple>
#include "block.h"

namespace hicma {

  // Used in polymorphic code to know what actually a class is.
  enum {
    HICMA_NODE,
    HICMA_HIERARCHICAL,
    HICMA_LOWRANK,
    HICMA_DENSE
  };

  class Node {
  public:
    int i_abs;
    int j_abs;
    int level;
    block_map_t block_map;
    
    Node();
    Node(const int _i_abs, const int _j_abs, const int _level);

    Node(const Node& ref);

    virtual ~Node();

    virtual Node* clone() const;

    friend void swap(Node& first, Node& second);

    virtual const Node& operator=(const double a);

    virtual const Node& operator=(const Node& A);
    virtual const Node& operator=(Node&& A);

    virtual const Node& operator=(Block A);

    virtual Block operator+(const Node& B) const;
    virtual Block operator+(Block&& B) const;
    virtual const Node& operator+=(const Node& B);
    virtual const Node& operator+=(Block&& B);
    virtual Block operator-(const Node& B) const;
    virtual Block operator-(Block&& B) const;
    virtual const Node& operator-=(const Node& B);
    virtual const Node& operator-=(Block&& B);
    virtual Block operator*(const Node& B) const;
    virtual Block operator*(Block&& B) const;

    virtual const bool is(const int enum_id) const;

    virtual const char* is_string() const;

    virtual double norm() const;

    virtual void print() const;

    virtual void getrf();

    virtual void trsm(const Node& A, const char& uplo);

    virtual void gemm(const Node& A, const Node& B);

    const block_map_t& get_map(void) const;

    virtual void create_dense_block(std::vector<double> &data);

    virtual std::vector<Block> get_data(void) const;

    virtual bool has_block(const int i, const int j) const;

    virtual void single_process_split(const int proc_id);

    virtual void multi_process_split(void);
  };
}
#endif
