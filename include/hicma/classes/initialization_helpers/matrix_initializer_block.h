/**
 * @file matrix_initializer_block.h
 * @brief Include the `MatrixInitializerBlock` class
 *
 * @copyright Copyright (c) 2020
 */
#ifndef hicma_classes_initialization_helpers_matrix_initializer_block_h
#define hicma_classes_initialization_helpers_matrix_initializer_block_h

#include "hicma/definitions.h"
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

/**
 * @brief `MatrixInitializer` specialization initializing matrix elements from a
 * large `Dense` matrix instance
 */
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

  /**
   * @brief Construct a new `MatrixInitializerBlock` object
   *
   * @param A
   * Large `Dense` matrix from which elements are used to assign to submatrices.
   * @param admis
   * Admissibility in terms of distance from the diagonal of the matrix on the
   * current recursion level (for `PositionBasedAdmis`) or admissibility constant
   * (for `GeometryBasedAdmis`)
   * @param eps
   * Fixed error threshold used for approximating admissible submatrices.
   * @param rank
   * Fixed rank to be used for approximating admissible submatrices.
   * @param params
   * Vector containing the underlying geometry information of the input `Dense` matrix
   * @param admis_type
   * Either `PositionBasedAdmis` or `GeometryBasedAdmis`
   */
  MatrixInitializerBlock(
    Dense&& A, double admis, double eps, int64_t rank,
    std::vector<std::vector<double>> params, AdmisType admis_type
  );

  /**
   * @brief Specialization for assigning matrix elements
   *
   * @param A
   * Matrix whose elements are to be assigned.
   * @param row_range
   * Row range of \p A. The start of the `IndexRange` is that within the root
   * level `Hierarchical` matrix.
   * @param col_range
   * Column range of \p A. The start of the `IndexRange` is that within the root
   * level `Hierarchical` matrix.
   *
   * Use the large `Dense` matrix stored in this class to assign elements. The
   * \p row_range and \p col_range are both used as indices into the large
   * `Dense` matrix that was passed to the constructor of this class.
   */
  void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const override;

};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_matrix_initializer_block_h
