#ifndef hicma_classes_initialization_helpers_matrix_initializer_h
#define hicma_classes_initialization_helpers_matrix_initializer_h

#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <tuple>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class ClusterTree;

class MatrixInitializer {
 private:
  int64_t admis;
  int64_t rank;

 public:
  // Special member functions
  MatrixInitializer() = delete;

  ~MatrixInitializer() = default;

  MatrixInitializer(const MatrixInitializer& A) = delete;

  MatrixInitializer& operator=(const MatrixInitializer& A) = delete;

  MatrixInitializer(MatrixInitializer&& A) = delete;

  MatrixInitializer& operator=(MatrixInitializer&& A) = default;

  // Additional constructors
  MatrixInitializer(int64_t admis, int64_t rank);

  virtual void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const = 0;

  Dense get_dense_representation(const ClusterTree& node) const;

  LowRank get_compressed_representation(const ClusterTree& node);

  bool is_admissible(const ClusterTree& node) const;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_initializer_h
