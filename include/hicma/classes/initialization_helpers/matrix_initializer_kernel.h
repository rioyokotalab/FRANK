/**
 * @file matrix_initializer_kernel.h
 * @brief Include the `MatrixInitializerKernel` class
 *
 * @copyright Copyright (c) 2020
 */
#ifndef hicma_classes_initialization_helpers_matrix_initializer_kernel_h
#define hicma_classes_initialization_helpers_matrix_initializer_kernel_h

#include "hicma/definitions.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include <cstdint>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class ClusterTree;
//class Dense;
class IndexRange;

/**
 * @brief `MatrixInitializer` specialization initializing matrix elements from a
 * kernel and parameters
 */
class MatrixInitializerKernel : public MatrixInitializer {
 private:
  void (*kernel)(
    double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const std::vector<std::vector<double>>& params,
    int64_t row_start, int64_t col_start
  ) = nullptr;
 public:

  // Special member functions
  MatrixInitializerKernel() = delete;

  ~MatrixInitializerKernel() = default;

  MatrixInitializerKernel(const MatrixInitializerKernel& A) = delete;

  MatrixInitializerKernel& operator=(const MatrixInitializerKernel& A) = delete;

  MatrixInitializerKernel(MatrixInitializerKernel&& A) = delete;

  MatrixInitializerKernel& operator=(MatrixInitializerKernel&& A) = delete;

  /**
   * @brief Construct a new `MatrixInitializerKernel` object
   *
   * @param kernel
   * Kernel to be used to assign matrix elements.
   * @param params
   * Vector with parameters used as input to the kernel.
   * @param admis
   * Distance-to-diagonal or standard admissibility condition constant.
   * @param rank
   * Fixed rank to be used for approximating admissible submatrices.
   * @param admis_type
   * Either POSITION_BASED_ADMIS or GEOMETRY_BASED_ADMIS
   */
  MatrixInitializerKernel(
    void (*kernel)(
      double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
      const std::vector<std::vector<double>>& params,
      int64_t row_start, int64_t col_start
    ),
    std::vector<std::vector<double>> params,
    double admis, int64_t rank, int admis_type
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
   * Uses the kernel and parameters stored in this class to assign elements. The
   * \p row_range and \p col_range are both used as indices into the vector of
   * parameters passed to the constructor of this class.
   */
  void fill_dense_representation(
    Dense<double>& A, const IndexRange& row_range, const IndexRange& col_range
  ) const override;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_initializer_kernel_h
