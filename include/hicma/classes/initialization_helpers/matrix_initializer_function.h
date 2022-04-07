/**
 * @file matrix_initializer_function.h
 * @brief Include the `MatrixInitializerFunction` class
 *
 * @copyright Copyright (c) 2020
 */
#ifndef hicma_classes_initialization_helpers_matrix_initializer_function_h
#define hicma_classes_initialization_helpers_matrix_initializer_function_h

#include "hicma/definitions.h"
#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include <cstdint>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

/**
 * @brief `MatrixInitializer` specialization initializing matrix elements from a
 * kernel and parameters
 */
template<typename T = double>
class MatrixInitializerFunction : public MatrixInitializer {
 private:
  void (*func)(
    T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const vec2d<double>& params,
    int64_t row_start, int64_t col_start
  ) = nullptr;
 
 public:

  // Special member functions
  MatrixInitializerFunction() = delete;

  ~MatrixInitializerFunction() = default;

  MatrixInitializerFunction(const MatrixInitializerFunction& A) = delete;

  MatrixInitializerFunction& operator=(const MatrixInitializerFunction& A) = delete;

  MatrixInitializerFunction(MatrixInitializerFunction&& A) = delete;

  MatrixInitializerFunction& operator=(MatrixInitializerFunction&& A) = delete;

  /**
   * @brief Construct a new `MatrixInitializerFunction` object
   *
   * @param func
   * Function to be used to assign matrix elements.
   * @param params
   * Vector with parameters used as input to the kernel.
   * @param admis
   * Distance-to-diagonal or standard admissibility condition constant.
   * @param rank
   * Fixed rank to be used for approximating admissible submatrices.
   * @param admis_type
   * Either POSITION_BASED_ADMIS or GEOMETRY_BASED_ADMIS
   */
  MatrixInitializerFunction(
    void (*func)(
      T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
      const vec2d<double>& params,
      int64_t row_start, int64_t col_start
    ),
    double admis, int64_t rank, int admis_type=POSITION_BASED_ADMIS,
    vec2d<double> params = vec2d<double>()
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
    Matrix& A, const IndexRange& row_range, const IndexRange& col_range
  ) const override;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_initializer_kernel_h
