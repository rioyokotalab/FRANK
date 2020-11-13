#ifndef hicma_operations_BLAS_h
#define hicma_operations_BLAS_h

#include "hicma/classes/dense.h"


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

/**
 * @brief Perform in-place matrix-matrix multiplication
 *
 * @param A
 * `Matrix` instance
 * @param B
 * `Matrix` instance
 * @param C
 * `Matrix` instance
 * @param alpha
 * Scalar value
 * @param beta
 * Scaalr value
 * @param transA
 * \p true if \p transpose(A) will be used, \p false otherwise
 * @param transB
 * \p true if \p transpose(B) will be used, \p false otherwise
 *
 * This function performs the matrix-matrix operation
 *
 * <tt>C = alpha*op(A)*op(B) + beta*C</tt>
 *
 * where <tt>op(X) = X</tt> or <tt>op(X) = transpose(X)</tt>.
 * Prior to calling, it is assumed that \p A, \p B, and \p C have been initialized with proper dimensions.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 *
 * <b>Note</b> that for any types of \p A, \p B, and \p C, at the lowest level it will end up with multiplication involving only `Dense` matrices.
 * For that operation, this method is made to behave like <a target="_blank" href="http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html#gaeda3cbd99c8fb834a60a6412878226e1"><tt>dgemm</tt></a> subroutine of BLAS/LAPACK. See the documentation for more information.
 */
void gemm(
  const Matrix& A, const Matrix& B, Matrix& C,
  double alpha=1, double beta=1,
  bool TransA=false, bool TransB=false
);

/**
 * @brief Perform matrix-matrix multiplication
 *
 * @param A
 * `Matrix` instance
 * @param B
 * `Matrix` instance
 * @param alpha
 * Scalar value
 * @param transA
 * \p true if \p transpose(A) will be transposed, \p false otherwise
 * @param transB
 * \p true if \p transpose(B) will be transposed, \p false otherwise
 *
 * @return `MatrixProxy`
 * Instance of `MatrixProxy` that owns the resulting matrix
 *
 * This function performs the matrix-matrix operation
 *
 * <tt>C = alpha*op(A)*op(B)</tt>
 *
 * where <tt>op(X) = X</tt> or <tt>op(X) = transpose(X)</tt>.
 * The type of \p C is determined dynamically based on the types of \p A and \p B.
 * Prior to calling, it is assumed that \p A and \p B have been initialized with proper dimensions.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 *
 * <b>Note</b> that for any types of \p A, \p B, and \p C, at the lowest level it will end up with multiplication involving only `Dense` matrices.
 * For that operation, this method is made to behave like <a target="_blank" href="http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html#gaeda3cbd99c8fb834a60a6412878226e1"><tt>dgemm</tt></a> subroutine of BLAS/LAPACK. See the documentation for more information.
 */
MatrixProxy gemm(
  const Matrix& A, const Matrix& B,
  double alpha=1,
  bool TransA=false, bool TransB=false
);

enum { TRSM_UPPER, TRSM_LOWER };
enum { TRSM_LEFT, TRSM_RIGHT };

/**
 * @brief Perform in-place triangular matrix multiplication
 *
 * @param A
 * A (lower or upper) triangular `Matrix`
 * @param B
 * `Matrix` instance
 * @param side
 * \p 'l' if \p A is multiplied from the left of \p B, \p 'r' if multiplied from the right
 * @param uplo
 * \p 'u' if \p A is an upper triangular matrix, \p 'l' if lower triangular
 * @param trans
 * \p 't' if \p transpose(A) will be used, \p 'n' otherwise
 * @param diag
 * \p 'u' if \p A is assumed to be unit triangular, \p 'n' otherwise
 * @param alpha
 * Scalar value
 *
 * This function performs in-place triangular matrix multiplication
 *
 * <tt>B = alpha*A*B</tt> or <tt>B = alpha*B*A</tt>,
 *
 * where \p A is an upper (or lower) triangular matrix and \p B is any matrix with proper size.
 * Prior to calling, it is assumed that \p A and \p B have been initialized with proper dimensions.
 * On finish, the result is overwritten on \p B.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 *
 * <b>Note</b> that for any types of \p A and \p B, at the lowest level it will end up with operations involving only `Dense` matrices.
 * Thus at the core, this method relies on <a target="_blank" href="http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaf07edfbb2d2077687522652c9e283e1e.html#gaf07edfbb2d2077687522652c9e283e1e"><tt>dtrmm</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation for more information.
 */
void trmm(
  const Matrix& A, Matrix& B,
  const char& side, const char& uplo, const char& trans, const char& diag,
  double alpha
);

void trmm(
  const Matrix& A, Matrix& B,
  const char& side, const char& uplo,
  double alpha
);

/**
 * @brief Solve triangular system of equations represented by matrices
 *
 * @param A
 * A (lower or upper) triangular `Matrix`
 * @param B
 * `Matrix` instance
 * @param uplo
 * \p TRSM_UPPER if \p A is an upper triangular matrix, \p TRSM_LOWER if lower triangular
 * @param lr
 * \p TRSM_LEFT if \p A is multiplied from the left of \p X, \p TRSM_RIGHT if multiplied from the right of \p X
 *
 * This function solves triangular matrix equation, i.e.
 *
 * Find \p X in <tt>A*X = B</tt> or <tt>X*A = B</tt>,
 *
 * where \p A is an upper (or lower) triangular matrix and \p B is any matrix with proper size.
 * Prior to calling, it is assumed that \p A and \p B have been initialized with proper dimensions.
 * On finish, the matrix \p X is overwritten on \p B.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 *
 * <b>Note</b> that for any types of \p A and \p B, at the lowest level it will end up with operations involving only `Dense` matrices.
 * Thus at the core, this method relies on <a target="_blank" href="http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaf07edfbb2d2077687522652c9e283e1e.html#gaf07edfbb2d2077687522652c9e283e1e"><tt>dtrsm</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation for more information.
 */
void trsm(const Matrix& A, Matrix& B, int uplo, int lr=TRSM_LEFT);

} // namespace hicma

#endif // hicma_operations_BLAS_h
