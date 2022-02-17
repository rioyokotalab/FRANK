/**
 * @file matrix_initializer_block.h
 * @brief Include the `MatrixInitializerBlock` class
 *
 * @copyright Copyright (c) 2020
 */
#ifndef hicma_classes_initialization_helpers_matrix_initializer_block_h
#define hicma_classes_initialization_helpers_matrix_initializer_block_h

#include "hicma/classes/initialization_helpers/matrix_initializer.h"


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

/**
 * @brief `MatrixInitializer` specialization initializing matrix elements from a
 * large `Dense` matrix instance
 */
template<typename U = double>
class MatrixInitializerBlock : public MatrixInitializer {
  private:
    Dense<U> matrix;
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
     * Distance-to-diagonal admissibility condition.
     * @param rank
     * Fixed rank to be used for approximating admissible submatrices.
     */
    MatrixInitializerBlock(Dense<U>&& A, double admis, int64_t rank);

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
      Matrix& A, const IndexRange& row_range, const IndexRange& col_range
    ) const override;

};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_matrix_initializer_block_h
