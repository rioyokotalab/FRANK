#ifndef hicma_operations_misc_h
#define hicma_operations_misc_h

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/matrix_proxy.h"

#include <cstdint>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

/**
 * @brief Get the number of rows of a matrix
 *
 * @param A
 * `Matrix` instance
 *
 * @return int64_t
 * Number of rows of data that is held by \p A
 *
 * This method returns the number of actual rows of floating points that is held by the \p A.
 * If necessary, this method will recurse deep inside a `Hierarchical` object to determine the total number
 * of rows of floating points that it possesses, which is done by accumulating rows from `Dense` and `LowRank`
 * objects that comprises the `Hierarchical` object.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 */
int64_t get_n_rows(const Matrix&);

/**
 * @brief Get the number of columns of a matrix
 *
 * @param A
 * `Matrix` instance
 *
 * @return int64_t
 * Number of columns of data that is held by \p A
 *
 * This method returns the number of actual columns of floating points that is held by the \p A.
 * If necessary, this method will recurse deep inside a `Hierarchical` object to determine the total number
 * of columns of floating points that it possesses, which is done by accumulating columns from `Dense` and `LowRank`
 * objects that comprises the `Hierarchical` object.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 */
int64_t get_n_cols(const Matrix&);

/**
 * @brief Calculate the condition number of a `Dense` matrix
 *
 * @param A
 * `Dense` matrix
 *
 * @return double
 * Condition number of \p A
 */
double cond(Dense<double> A);

void sortByMortonIndex(std::vector<std::vector<double>> &x, int64_t level, std::vector<int64_t>& perm);

/**
 * @brief Generate equally spaced floating points between a specified range
 *
 * @param N
 * Number of numbers to be generated
 * @param minVal
 * Minimum value of the range
 * @param maxVal
 * Maximum value of the range
 *
 * @return std::vector<double>
 * Vector containing \p N equally spaced numbers taken from the interval [\p minVal, \p maxVal].
 */
std::vector<double> equallySpacedVector(
  int64_t N, double minVal, double maxVal);

/**
 * @brief Split a matrix based on the specified number of block-row and block-column
 *
 * @param A
 * `Matrix` to be split
 * @param n_row_blocks
 * Desired number of block-rows
 * @param n_col_blocks
 * Desired number of block-columns
 * @param copy
 * Determines whether to perform deep copy of \p A or no (share the underlying data)
 *
 * @return `Hierarchical` object containing `n_row_blocks`-by-`n_col_blocks` blocks
 *
 * This method subdivide a given matrix \p A by row and column.
 * If \p copy is set to \p false, this method doesn't perform deep copy of \p A and instead uses shared pointer
 * so that the resulting `Hierarchical` object points to the same data as \p A.
 * Otherwise, it creates a deep copy of \p A that will be owned by the resulting `Hierarchical` object.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 */
Hierarchical split(
  const Matrix& A, int64_t n_row_blocks, int64_t n_col_blocks, bool copy=false
);

/**
 * @brief Split a matrix based on an existing subdivision of a `Hierarchical` object
 *
 * @param A
 * `Matrix` to be split
 * @param like
 * `Hierarchical` object whose subdivision pattern will be used
 * @param copy
 * Determines whether to perform deep copy of \p A or no (share the underlying data)
 *
 * @return `Hierarchical` object containing \p A that has been subdivided based on the structure of \p like
 *
 * This method subdivide a given `Matrix` \p A using the same block subdivision pattern of \p like,
 * resulting in a `Hierarchical` object with the same structure as \p like but has the data from \p A.
 * If \p copy is set to \p false, this method won't perform deep copy and instead uses shared pointer
 * so that the resulting `Hierarchical` object points to the same data as \p A.
 * Otherwise, it performs deep copy of A that will be owned by the resulting `Hierarchical` object.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 */
Hierarchical split(const Matrix& A, const Hierarchical& like, bool copy=false);

MatrixProxy shallow_copy(const Matrix& A);

/**
 * @brief Compute the L2 norm of a matrix
 *
 * @param A
 * `Matrix` instance
 *
 * @return double
 * L2 norm of \p A
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 */
double norm(const Matrix&);

/**
 * @brief Compute the transpose of a matrix
 *
 * @param A
 * `Matrix` whose tranpose will be computed
 *
 * @return Instance of `MatrixProxy` that owns transpose of \p A
 *
 * Note that the transpose of \p A that is held by the return value is obtained from a copy of \p A, meaning that \p A itself is not modified.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 */
MatrixProxy transpose(const Matrix&);

/**
 * @brief Resize a matrix into the specified dimension
 *
 * @param A
 * Matrix to be resized
 * @param n_rows
 * New row dimension less than or equal to the current row dimension of \p A
 * @param n_cols
 * New column dimension less than or equal to the current row dimension of \p A
 *
 * @return Instance of `MatrixProxy` object that owns the resized \p A
 *
 * Note that the resized \p A that is held by the return value is obtained from a copy of \p A, meaning that \p A itself is not modified.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 */
MatrixProxy resize(const Matrix&, int64_t n_rows, int64_t n_cols);

} // namespace hicma

#endif // hicma_operations_misc_h
