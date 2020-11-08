#ifndef hicma_classes_initialization_helpers_matrix_initializer_block_h
#define hicma_classes_initialization_helpers_matrix_initializer_block_h

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include <cstdint>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class ClusterTree;
class IndexRange;

class MatrixInitializerBlock : public MatrixInitializer {
 private:
  Dense matrix;
 public:
  // Special member functions
  MatrixInitializerBlock() = delete;

  ~MatrixInitializerBlock() = default;

  MatrixInitializerBlock(const MatrixInitializerBlock& A) = delete;

  MatrixInitializerBlock& operator=(const MatrixInitializerBlock& A) = delete;

  MatrixInitializerBlock(MatrixInitializerBlock&& A) = delete;

  MatrixInitializerBlock& operator=(MatrixInitializerBlock&& A) = default;

  // Additional constructors
  MatrixInitializerBlock(Dense&& A, int64_t admis, int64_t rank);

  // Utility methods
  void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const override;

};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_matrix_initializer_block_h
