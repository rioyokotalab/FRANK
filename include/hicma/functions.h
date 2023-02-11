#ifndef hicma_functions_h
#define hicma_functions_h

#include <cstdint>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

/**
 * @brief Kernel function that generates zero matrix
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information (if applicable)
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void zeros(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

/**
 * @brief Kernel function that generates identity matrix
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information (if applicable)
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void identity(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

/**
 * @brief Kernel function that generates random matrix using normal distribution
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information (if applicable)
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * Entries of the matrix are random numbers generated using normal distribution.
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void random_normal(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

/**
 * @brief Kernel function that generates random matrix using uniform distribution
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information (if applicable)
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * Entries of the matrix are random numbers generated using uniform distribution from the range [0,1).
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void random_uniform(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

/**
 * @brief Kernel function that generates matrix with integer entries within the specified range
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information (if applicable)
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * Entries of the matrix are integers from 0 to A_rows*A_cols, such that the (i,j)-th entry is given by:
 * <tt>i*A_cols + j</tt>
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void arange(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

/**
 * @brief Kernel function that generates cauchy matrix using set of 2D points
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information: list of 2D coordinates
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void cauchy2d(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

/**
 * @brief Kernel function for N-dimensional laplacian
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information: list of N-dimensional coordinates
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void laplacend(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

/**
 * @brief Kernel function for N-dimensional helmholtz
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information: list of N-dimensional coordinates
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void helmholtznd(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

/**
 * @brief Kernel function that generates gaussian matrix using set of 2D points
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information: list of 2D coordinates
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void gaussiannd(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

/**
 * @brief Kernel function that generates IMQ matrix using set of 2D points
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information: list of 2D coordinates
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void imqnd(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

/**
 * @brief Kernel function that generates Matern matrix using set of 2D points
 *
 * @param A
 * Array to be filled with entries
 * @param A_rows
 * Number of rows of \p A
 * @param A_cols
 * Number of columns of \p A
 * @param A_stride
 * Stride of \p A
 * @param x
 * 2D vector that holds geometry information: list of 2D coordinates
 * @param row_start
 * Row offset (if generating a submatrix)
 * @param col_start
 * Column offset (if generating a submatrix)
 *
 * This function is used as kernel to generate matrix with `MatrixInitializerKernel`.
 */
template<typename T = double>
void maternnd(
  T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
  const std::vector<std::vector<double>>& x,
  int64_t row_start, int64_t col_start
);

} // namespace hicma

#endif // hicma_functions_h
