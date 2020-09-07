#ifndef hicma_classes_initialization_helpers_matrix_initializer_h
#define hicma_classes_initialization_helpers_matrix_initializer_h

#include "hicma/classes/dense.h"
// TODO Note that this include is only for the enum (NORMAL_BASIS)
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/initialization_helpers/basis_tracker.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class ClusterTree;

class MatrixInitializer {
 private:
  int64_t admis;
  int64_t rank;
  BasisTracker<IndexRange, NestedBasis> col_basis, row_basis;
  NestedTracker col_tracker, row_tracker;
  int basis_type = NORMAL_BASIS;

  void find_admissible_blocks(const ClusterTree& node);

  Dense make_block_row(const NestedTracker& tracker) const;
  void construct_nested_col_basis(NestedTracker& row_tracker);

  Dense make_block_col(const NestedTracker& tracker) const;
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
    int64_t admis, int64_t rank, int basis_type
  );

  // Utility methods
  virtual void fill_dense_representation(
    Dense& A, const ClusterTree& node
  ) const = 0;

  virtual void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const = 0;

  virtual Dense get_dense_representation(const ClusterTree& node) const = 0;

  LowRank get_compressed_representation(const ClusterTree& node);

  void create_nested_basis(const ClusterTree& node);

  bool is_admissible(const ClusterTree& node) const;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_initializer_h
