#ifndef hicma_classes_initialization_helpers_matrix_initializer_h
#define hicma_classes_initialization_helpers_matrix_initializer_h

#include "hicma/classes/dense.h"
// TODO Note that this include is only for the enum (NORMAL_BASIS)
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/intitialization_helpers/basis_tracker.h"
#include "hicma/classes/intitialization_helpers/index_range.h"

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class ClusterTree;

class MatrixInitializer {
 private:
  void (*kernel)(
    Dense& A,
    const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ) = nullptr;
  const std::vector<std::vector<double>>& x;
  int64_t admis;
  int64_t rank;
  BasisTracker<IndexRange> col_basis, row_basis;
  NestedTracker col_tracker, row_tracker;
  int basis_type = NORMAL_BASIS;

  void find_admissible_blocks(const ClusterTree& node);

  void construct_nested_col_basis(NestedTracker& row_tracker);
  void construct_nested_row_basis(NestedTracker& col_tracker);
 public:

  // Special member functions
  MatrixInitializer() = delete;

  ~MatrixInitializer() = default;

  MatrixInitializer(const MatrixInitializer& A) = delete;

  MatrixInitializer& operator=(const MatrixInitializer& A) = delete;

  MatrixInitializer(MatrixInitializer&& A) = delete;

  MatrixInitializer& operator=(MatrixInitializer&& A) = default;

  // Additional constructors
  MatrixInitializer(
    void (*kernel)(
      Dense& A, const std::vector<std::vector<double>>& x,
      int64_t row_start, int64_t col_start
    ),
    const std::vector<std::vector<double>>& x,
    int64_t admis, int64_t rank,
    int basis_type
  );

  // Utility methods
  virtual void fill_dense_representation(
    Dense& A, const ClusterTree& node
  ) const;

  virtual void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const;

  virtual Dense get_dense_representation(const ClusterTree& node) const;

  Dense make_block_row(const NestedTracker& tracker) const;

  Dense make_block_col(const NestedTracker& tracker) const;

  virtual LowRank get_compressed_representation(const ClusterTree& node);

  void register_admissible_block(const ClusterTree& node);

  void create_nested_basis(const ClusterTree& node);

  bool is_admissible(const ClusterTree& node) const;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_initializer_h
