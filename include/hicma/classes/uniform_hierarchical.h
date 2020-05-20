#ifndef hicma_classes_uniform_hierarchical_h
#define hicma_classes_uniform_hierarchical_h

#include "hicma/classes/hierarchical.h"

#include <cstdint>
#include <memory>
#include <vector>


namespace hicma
{

class Dense;
class LowRankShared;
class ClusterTree;
class MatrixInitializer;
class MatrixProxy;

class UniformHierarchical : public Hierarchical {
 private:
  std::vector<std::shared_ptr<Dense>> col_basis, row_basis;

  Dense make_block_row(
    const ClusterTree& node,
    const MatrixInitializer& initer,
    int64_t admis
  );

  Dense make_block_col(
    const ClusterTree& node,
    const MatrixInitializer& initer,
    int64_t admis
  );

  LowRankShared construct_shared_block_id(
    const ClusterTree& node,
    const MatrixInitializer& initer,
    std::vector<std::vector<int64_t>>& selected_rows,
    std::vector<std::vector<int64_t>>& selected_cols,
    int64_t rank,
    int64_t admis
  );

  LowRankShared construct_shared_block_svd(
    const ClusterTree& node,
    const MatrixInitializer& initer,
    int64_t rank,
    int64_t admis
  );
 public:
  // Special member functions
  UniformHierarchical() = default;

  ~UniformHierarchical() = default;

  UniformHierarchical(const UniformHierarchical& A);

  UniformHierarchical& operator=(const UniformHierarchical& A) = default;

  UniformHierarchical(UniformHierarchical&& A) = default;

  UniformHierarchical& operator=(UniformHierarchical&& A) = default;

  // Conversion constructors
  UniformHierarchical(MatrixProxy&&);

  // Additional constructors
  UniformHierarchical(int64_t n_row_blocks, int64_t n_col_blocks);

  UniformHierarchical(
    const ClusterTree& node,
    const MatrixInitializer& initer,
    int64_t rank,
    int64_t admis=1,
    bool use_svd=false
  );

  UniformHierarchical(
    void (*func)(
      Dense& A, const std::vector<std::vector<double>>& x,
      int64_t row_start, int64_t col_start
    ),
    const std::vector<std::vector<double>>& x,
    int64_t n_row, int64_t n_cols,
    int64_t rank,
    int64_t nleaf,
    int64_t admis=1,
    int64_t n_row_blocks=2, int64_t n_col_blocks=2,
    bool use_svd=false,
    int64_t row_start=0, int64_t col_start=0
  );

  // Additional indexing methods
  Dense& get_row_basis(int64_t i);

  const Dense& get_row_basis(int64_t i) const;

  Dense& get_col_basis(int64_t j);

  const Dense& get_col_basis(int64_t j) const;

  // Utiliry methods
  void copy_col_basis(const UniformHierarchical& A);

  void copy_row_basis(const UniformHierarchical& A);

  void set_col_basis(int64_t i, int64_t j);

  void set_row_basis(int64_t i, int64_t j);
};

} // namespace hicma

#endif // hicma_classes_uniform_hierarchical_h
