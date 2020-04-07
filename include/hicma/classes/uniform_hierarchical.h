#ifndef hicma_classes_uniform_hierarchical_h
#define hicma_classes_uniform_hierarchical_h

#include "hicma/classes/hierarchical.h"

#include "yorel/yomm2/cute.hpp"

#include <memory>
#include <vector>


namespace hicma
{

class Dense;
class IndexRange;
class LowRankShared;
class NodeProxy;

class UniformHierarchical : public Hierarchical {
 private:
  std::vector<std::shared_ptr<Dense>> col_basis, row_basis;

  Dense make_block_row(
    int row, int i_abs, int j_abs,
    void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
    std::vector<double>& x,
    int admis,
    int i_begin, int j_begin
  );

  Dense make_block_col(
    int col, int i_abs, int j_abs,
    void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
    std::vector<double>& x,
    int admis,
    int i_begin, int j_begin
  );

  LowRankShared construct_shared_block_id(
    int i, int j, int i_abs, int j_abs,
    std::vector<std::vector<int>>& selected_rows,
    std::vector<std::vector<int>>& selected_cols,
    void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
    std::vector<double>& x,
    int rank,
    int admis,
    int i_begin, int j_begin
  );

  LowRankShared construct_shared_block_svd(
    int i, int j, int i_abs, int j_abs,
    void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
    std::vector<double>& x,
    int rank,
    int admis,
    int i_begin, int j_begin
  );
 public:
  // Special member functions
  UniformHierarchical() = default;

  ~UniformHierarchical() = default;

  UniformHierarchical(const UniformHierarchical& A);

  UniformHierarchical& operator=(const UniformHierarchical& A) = default;

  UniformHierarchical(UniformHierarchical&& A) = default;

  UniformHierarchical& operator=(UniformHierarchical&& A) = default;

  // Overridden functions from Hierarchical
  std::unique_ptr<Node> clone() const override;

  std::unique_ptr<Node> move_clone() override;

  const char* type() const override;

  // Conversion constructors
  UniformHierarchical(NodeProxy&&);

  // Additional constructors
  UniformHierarchical(int ni_level, int nj_level);

  UniformHierarchical(
    IndexRange row_range, IndexRange col_range,
    void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
    std::vector<double>& x,
    int rank,
    int nleaf,
    int admis=1,
    int ni_level=2, int nj_level=2,
    bool use_svd=false,
    int i_begin=0, int j_begin=0,
    int i_abs=0, int j_abs=0
  );

  UniformHierarchical(
    void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
    std::vector<double>& x,
    int ni, int nj,
    int rank,
    int nleaf,
    int admis=1,
    int ni_level=2, int nj_level=2,
    bool use_svd=false,
    int i_begin=0, int j_begin=0,
    int i_abs=0, int j_abs=0
  );

  // Additional indexing methods
  Dense& get_row_basis(int i);

  const Dense& get_row_basis(int i) const;

  Dense& get_col_basis(int j);

  const Dense& get_col_basis(int j) const;

  // Utiliry methods
  void copy_col_basis(const UniformHierarchical& A);

  void copy_row_basis(const UniformHierarchical& A);

  void set_col_basis(int i, int j);

  void set_row_basis(int i, int j);
};

register_class(UniformHierarchical, Hierarchical)

} // namespace hicma

#endif // hicma_classes_uniform_hierarchical_h
