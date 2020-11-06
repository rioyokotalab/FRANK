/**
 * @file matrix.h
 * @brief Include the base `Matrix` class.
 *
 * @copyright Copyright (c) 2020
 */
#ifndef hicma_classes_matrix_h
#define hicma_classes_matrix_h


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

/**
 * @brief Abstract matrix class. All other matrix types derive from this.
 *
 * This abstract matrix class has no properties, holds no data and no operations
 * are defined for it. It serves only as a parent class for all other matrix
 * classes. Having a common parent class is necessary for \OMMs, a central part
 * the HiCMA library.
 */
class Matrix {
 public:
  // Special member functions
  Matrix() = default;

  virtual ~Matrix() = default;

  Matrix(const Matrix& A) = default;

  Matrix& operator=(const Matrix& A) = default;

  Matrix(Matrix&& A) = default;

  Matrix& operator=(Matrix&& A) = default;
};

} // namespace hicma

#endif // hicma_classes_matrix_h
