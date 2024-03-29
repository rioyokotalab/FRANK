#ifndef FRANK_operations_misc_h
#define FRANK_operations_misc_h

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/matrix_proxy.h"

#include <cstdint>
#include <vector>


/**
 * @brief General namespace of the FRANK library
 */
namespace FRANK
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
 * Read \ext_FRANK for more information.
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
 * Read \ext_FRANK for more information.
 */
int64_t get_n_cols(const Matrix&);

/**
 * @brief Get subset of columns from a `Dense` matrix
 *
 * @param A
 * M-by-N `Dense` matrix
 * @param Pr
 * Column indices
 *
 * @return `Dense`
 * A `Dense` matrix containing the specified subset of columns ordered as \p Pr.
 */
Dense get_cols(const Dense& A, std::vector<int64_t> Pr);

/**
 * @brief Reset all elements to zero
 *
 * @param A
 * `Matrix` instance. Modified on finish
 *
 * This method reset all elements of \p A to zero.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_FRANK for more information.
 */
void zero_all(Matrix& A);

/**
 * @brief Reset elements below the main diagonal to zero
 *
 * @param A
 * `Matrix` instance. Modified on finish
 *
 * This method set all elements below the main diagonal of \p A to zero.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_FRANK for more information.
 */
void zero_lower(Matrix& A);

/**
 * @brief Reset elements above the main diagonal to zero
 *
 * @param A
 * `Matrix` instance. Modified on finish
 *
 * This method set all elements above the main diagonal of \p A to zero.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_FRANK for more information.
 */
void zero_upper(Matrix& A);

/**
 * @brief Calculate the condition number of a `Dense` matrix
 *
 * @param A
 * `Dense` matrix
 *
 * @return double
 * Condition number of \p A with respect to 2-norm
 */
double cond(Dense A);

/**
 * @brief Find SVD compression rank based on relative error threshold
 *
 * @param S
 * `Dense` matrix containing singular values in its diagonal elements
 * @param eps
 * Relative error threshold of the compression
 *
 * Find truncation rank r from a diagonal matrix containing singular values such that r is the smallest value that satisfies
 *
 * <tt>|A-A_r|_F < eps*|A|_F</tt>.
 *
 * where |A|_F denotes the Frobenius norm of A.
 */
int64_t find_svd_truncation_rank(const Dense& S, const double eps);

/**
 * @brief Sort n-dimensional points on euclidean space based on Morton Ordering
 *
 * @param x
 * Points to be sorted
 * @param level
 * Recursion level of the morton quadtree
 * @param perm
 * Row (and column) permutation for dense matrix that uses the original ordering
 *
 * The parameter \p level controls the depth of quadtree, i.e. the number of different boxes that will be created is 2<sup>level+1</sup>.
 *
 * The n-dimensional points is assumed to be stored as follows
 * ```
 * axis\point   p1  p2  p3 ... pn
 *   x          x1  x2  x3 ... xn
 *   y          y1  y2  y3 ... yn
 *   z          z1  z2  z3 ... zn
 * ...
 * ```
 * i.e. each row of x (`x[i]`) contain the coordinates along one axis.
 */
void sortByMortonIndex(std::vector<std::vector<double>> &x, const int64_t level, std::vector<int64_t>& perm);

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
  const int64_t N, const double minVal, const double maxVal);

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
 * Read \ext_FRANK for more information.
 */
Hierarchical split(
  const Matrix& A, const int64_t n_row_blocks, const int64_t n_col_blocks, const bool copy=false
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
 * Read \ext_FRANK for more information.
 */
Hierarchical split(const Matrix& A, const Hierarchical& like, const bool copy=false);

/**
 * @brief Get the shallow copy of a general `Matrix`
 *
 * @param A
 * `Matrix` instance
 *
 * @return
 * `Matrix` instance that points to the same data as \p A
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_FRANK for more information.
 */
MatrixProxy shallow_copy(const Matrix& A);

/**
 * @brief Compute the L2 (Frobenius) norm of a matrix
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
 * Read \ext_FRANK for more information.
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
 * Read \ext_FRANK for more information.
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
 * Read \ext_FRANK for more information.
 */
MatrixProxy resize(const Matrix&, const int64_t n_rows, const int64_t n_cols);

} // namespace FRANK

#endif // FRANK_operations_misc_h
