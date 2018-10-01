#ifndef hierarchical_h
#define hierarchical_h
#include "node.h"
#include "block.h"
#include <vector>

namespace hicma {
  class Hierarchical : public Node {
  public:
    int dim[2];
    std::vector<Block> data;

    Hierarchical();

    Hierarchical(const int m);

    Hierarchical(const int m, const int n);

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

    Hierarchical(const int M, const int N, const int MB, const int NB,
                 const int P, const int Q, MPI_Comm mpi_comm);

    Hierarchical(const Hierarchical& A);
    Hierarchical(Hierarchical&& A);

    Hierarchical(const Hierarchical* A);

    Hierarchical(const Block& A);

    Hierarchical* clone() const override;

    friend void swap(Hierarchical& first, Hierarchical& second);

    const Node& operator=(const Node& A) override;
    const Node& operator=(Node&& A) override;
    const Hierarchical& operator=(Hierarchical A);

    const Node& operator=(Block A) override;

    const Node& operator=(const double a) override;

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

    const Node& operator[](const int i) const;
    Block& operator[](const int i);

    const Node& operator()(const int i, const int j) const;
    Block& operator()(const int i, const int j);

    const bool is(const int enum_id) const override;

    const char* is_string() const override;

    double norm() const override;

    void print() const override;

    void getrf() override;

    void trsm(const Node& A, const char& uplo) override;

    void gemm(const Node& A, const Node& B) override;

    void gemm_row(
        const Hierarchical& A,
        const Hierarchical& B,
        const int i, const int j, const int k_min, const int k_max);

    void create_dense_block(std::vector<double> &data) override;

    std::vector<Block> get_data(void) const override;

    bool has_block(const int i, const int j) const override;

    void single_process_split(const int proc_id) override;

    void multi_process_split(void) override;
  };
}
#endif
