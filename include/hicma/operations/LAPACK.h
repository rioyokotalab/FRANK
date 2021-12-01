#ifndef hicma_operations_LAPACK_h
#define hicma_operations_LAPACK_h

#include <cstdint>
#include <tuple>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;
class MatrixProxy;
class Dense;
class Hierarchical;

std::tuple<Dense, std::vector<int64_t>> geqp3(Matrix& A);

/**
 * @brief Compute full Householder QR factorization of a general matrix
 * using compact WY representation for \p Q
 *
 * @param A
 * M-by-N `Matrix` to be factorized. Overwritten on finish.
 * @param T
 * N-by-N `Matrix` that stores "T" matrix for the compact WY representation
 *
 * This function performs Householder QR factorization, using compact WY representation to implicitly store the resulting orthogonal factor \p Q.
 * Prior to calling, it is assumed that \p A and \p T have been initialized with proper dimensions.
 * Upon finish, \p A will be overwritten with lower trapezoidal matrix \p V and upper triangular matrix \p R.
 * <b>Currently only support \p A and \p T of type `Dense`</b>.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 *
 * <b>Note</b> that for any type of \p A, at the lowest level it will end up with operations involving only `Dense` matrices.
 * Thus at the core, this method relies on <a target="_blank" href="http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga3ad112f2b0890b3815e696628906f30c.html#ga3ad112f2b0890b3815e696628906f30c"><tt>dgeqrt3</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation for more information.
 */
void geqrt(Matrix& A, Matrix& T);

void geqrt2(Dense&, Dense&);

/**
 * @brief Compute reduced QR factorization of a general matrix
 * using Modified Gram-Schmidt iteration
 *
 * @param A
 * M-by-N `Matrix` to be factorized. Overwritten with \p Q on finish
 * @param R
 * N-by-N `Matrix` that is overwritten with upper triangular factor \p R on finish
 *
 * This function performs MGS QR factorization
 * Prior to calling, it is assumed that \p A and \p R have been initialized with proper dimensions.
 * Upon finish, \p A will be overwritten with the resulting orthogonal factor \p Q.
 */
void mgs_qr(Dense&, Dense&);

/**
 * @brief Compute LU factorization of a general matrix
 *
 * @param A
 * M-by-N `Matrix` to be factorized. Modified on finish.
 *
 * @return Tuple of `MatrixProxy` instances containing the lower and upper triangular factors
 *
 * This method performs LU decomposition of \p A without pivoting.
 * On finish, \p A becomes an empty object.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 *
 * <b>Note</b> that for any type of \p A, at the lowest level it will end up with operations involving only `Dense` matrices.
 * Thus at the core, this method relies on <a target="_blank" href="http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html#ga0019443faea08275ca60a734d0593e60"><tt>dgetrf</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation for more information.
 */
std::tuple<MatrixProxy, MatrixProxy> getrf(Matrix& A);

/**
 * @brief Compute one-sided interpolative decomposition (ID) of a `Dense` matrix
 *
 * @param A
 * M-by-N `Dense` instance to be factorized. Modified on finish.
 * @param k
 * Determines how many columns that needs to be taken (stopping criterion)
 *
 * @return TODO explain
 *
 * TODO add more explanation
 * This method performs one-sided interpolative decomposition of a given matrix.
 */
std::tuple<Dense, std::vector<int64_t>> one_sided_id(Matrix& A, int64_t k);

// TODO Does this need to be in the header?
Dense get_cols(const Dense& A, std::vector<int64_t> P);

/**
 * @brief Compute two-sided interpolative decomposition (ID) of a `Dense` matrix
 *
 * @param A
 * M-by-N `Dense` instance to be factorized. Modified on finish.
 * @param k
 * Determines how many columns that needs to be taken (stopping criterion)
 *
 * @return TODO explain
 *
 * TODO add more explanation
 * This method performs two-sided interpolative decomposition of a given matrix.
 */
std::tuple<Dense, Dense, Dense> id(Matrix& A, int64_t k);

/**
 * @brief Apply block householder reflector or its transpose to a general rectangular matrix
 *
 * @param V
 * M-by-N Lower trapezoidal `Matrix`
 * @param T
 * N-by-N Upper triangular `Matrix` of the compact WY representation
 * @param C
 * M-by-N rectangular `Matrix`
 * @param trans
 * \p true if applying the transpose of the reflector, \p false otherwise.
 *
 * This function applies a real block reflector \p H or \p transpose(H) to a real M-by-N rectangular matrix \p C, such that
 *
 * <tt>C = H*C or transpose(H)*C</tt>
 *
 * The block reflector \H is obtained from <tt>I-V*T*transpose(V)</tt> where \p I is identity matrix with proper dimension.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 *
 * <b>Only support \p V and \p T of type `Dense` at the moment</b>. Also <b>note</b> that for any combination of types, at the lowest level it will end up with operations involving only `Dense` matrices.
 * Thus at the core, this method relies on <a target="_blank" href="http://www.netlib.org/lapack/explore-html/d8/d9b/group__double_o_t_h_e_rauxiliary_ga83c81583bd444e0cf021fb006cd9a5e8.html#ga83c81583bd444e0cf021fb006cd9a5e8"><tt>dlarfb</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation for more information.
 */
void larfb(const Matrix& V, const Matrix& T, Matrix& C, bool trans);

/**
 * @brief Generate random matrix with specified singular values
 *
 * This method generates a random `Dense` matrix with specified singular values.
 * It relies on <a target="_blank" href="http://www.netlib.org/lapack/explore-html/d1/dc0/group__double__matgen_gadf4ba9c37cb5f67132e71433efa825d4.html#gadf4ba9c37cb5f67132e71433efa825d4"><tt>dlatms</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation page for detailed explanation about the parameters.
 */
void latms(
  const char& dist,
  std::vector<int>& iseed,
  const char& sym,
  std::vector<double>& d,
  int mode,
  double cond,
  double dmax,
  int kl, int ku,
  const char& pack,
  Dense& A
);

/**
 * @brief Compute QR factorization of a general matrix
 *
 * @param A
 * M-by-N `Matrix` to be factorized. Overwritten on finish
 * @param Q
 * M-by-N `Matrix` with the same type as \p A. On finish, filled with orthogonal factor
 * @param R
 * N-by-N `Matrix` with the same type as \p A. On finish, filled with upper triangular factor
 *
 * This method perform reduced QR factorization of a given matrix.
 * Prior to calling, \p Q and \p R need to be initialized with proper dimensions.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 */
void qr(Matrix& A, Matrix& Q, Matrix& R);

void orthogonalize_block_col(int64_t, const Matrix&, Matrix&, Matrix&);

/**
 * @brief Zero the lower triangular portion of a matrix
 *
 * @param A
 * `Matrix` instance. Modified on finish
 *
 * This method set the elements on lower triangular portion of \p A into zero.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 */
void zero_lowtri(Matrix& A);

/**
 * @brief Reset all elements of a matrix to zero
 *
 * @param A
 * `Matrix` instance. Modified on finish
 *
 * This method reset all elements of \p A into zero.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_hicma for more information.
 */
void zero_whole(Matrix& A);

void triangularize_block_col(int64_t, Hierarchical&, Hierarchical&);

void apply_block_col_householder(const Hierarchical&, const Hierarchical&, int64_t, bool, Hierarchical&, int64_t);

/**
 * @brief Perform RQ factorization of a `Dense` matrix
 *
 * @param A
 * M-by-N `Dense` matrix to be factorized. Modified on finish
 * @param R
 * M-by-M (or M-by-N) `Dense` matrix. On finish, filled with upper triangular (or upper trapezoidal) factor
 * @param Q
 * M-by-N (or N-by-N) `Dense` matrix. On finish, filled with orthogonal factor
 *
 * This method performs RQ factorization of a given `Dense` matrix.
 * Internally calls <a target="_blank" href="http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga7bba0d791b011eb5425ecbf500e9be2c.html#ga7bba0d791b011eb5425ecbf500e9be2c"><tt>dgerqf</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation page for more information.
 */
void rq(Matrix&, Matrix&, Matrix&);

/**
 * @brief Computes Singular Value Decomposition (SVD) of a `Dense` matrix
 *
 * @param A
 * M-by-N `Dense` matrix to be factorized. Modified on finish
 *
 * @return A tuple containing three `Dense` instances \p U, \p S, and \p V^T
 *
 * This method computes the SVD of a `Dense` matrix \p A, i.e.
 *
 * <tt>A = U*S*V</tt>
 *
 * such that \p U is M-by-k, \p S is k-by-k, and \p V is k-by-N matrices.
 * This method internally calls <a target="_blank" href="http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html#ga84fdf22a62b12ff364621e4713ce02f2"><tt>dgesvd</tt></a> subroutine provided by BLAS/LAPACK to perform the operation.
 * See the documentation page for more information.
 */
std::tuple<Dense, Dense, Dense> svd(Dense& A);

std::tuple<Dense, Dense, Dense> sdd(Dense& A);

// TODO Does this need to be in the header?
/**
 * @brief Compute the singular values of a `Dense` matrix
 *
 * @param A
 * M-by-N `Dense` matrix
 *
 * @return a vector containing singular values of \p A
 *
 * This method uses Singular Value Decomposition to compute the singular values of \p A.
 * The singular values are sorted from the largest to the smallest.
 */
std::vector<double> get_singular_values(Dense& A);

/**
 * @brief Apply a "lower trapezoidal" blocked reflector to a matrix composed of two blocks
 *
 * @param V
 * M-by-N Lower trapezoidal matrix of the compact WY representation of the block reflector
 * @param T
 * N-by-N Upper triangular matrix of the compact WY representation of the block reflector
 * @param A
 * K-by-N matrix
 * @param B
 * M-by-N matrix
 * @param trans
 * \p true if applying the transpose of the reflector, \p false otherwise.
 *
 * This function applies a "triangular pentagonal" blocked reflector to a matrix \p C such that
 *
 * <tt>C = H*C or transpose(H)*C</tt>
 *
 * where \p C is a matrix consisting two blocks, \p A on top of \p B.
 *
 * <b>Note</b> that for any type of \p V, \p T, \p A, and \p B, at the lowest level it will end up with operations involving only `Dense` matrices.
 * Thus this method relies on, and is made to behave like <a target="_blank" href="http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_gac998dca531aab64da39faff6b9dd9675.html#gac998dca531aab64da39faff6b9dd9675"><tt>dtpmqrt</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation page for more information
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (i.e. derivatives of `Matrix`) is implemented as a specialization of \OMM.
 * Dispatcher will select the correct implementation based on the types of parameters at runtime.
 * Read \ext_hicma for more information.
 */
void tpmqrt(const Matrix& V, const Matrix& T, Matrix& A, Matrix& B, bool);

/**
 * @brief Computes QR factorization of a "lower trapezoidal" matrix using compact WY representation for Q
 *
 * @param A
 * N-by-N upper triangular matrix. Modified on finish
 * @param B
 * M-by-N rectangular matrix. Modified on finish
 * @param T
 * N-by-N matrix. On finish, filled with upper triangular matrix for the compact WY representation
 *
 * This method computes the QR factorization of a lower trapezoidal (N+M)-by-N \p C, which consists of two blocks, \p A on top of \p B.
 * On finish, \p A is overwritten with the resulting upper triangular factor and the orthogonal factor \p Q is stored as compact WY representation
 * of "lower-trapezoidal" blocked reflector using rectangular matrix \p V that overwrites \p B and upper triangular matrix \p T.
 *
 * <b>Note</b> that for any type of \p A, \p B, and \p T, at the lowest level it will end up with operations involving only `Dense` matrices.
 * Thus this method relies on, and is made to behave like <a target="_blank" href="http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_gaa02cc2297f978edb5ef2a8fd1dcc9321.html#gaa02cc2297f978edb5ef2a8fd1dcc9321"><tt>dtpqrt</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation page for more information
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (i.e. derivatives of `Matrix`) is implemented as a specialization of \OMM.
 * Dispatcher will select the correct implementation based on the types of parameters at runtime.
 * Read \ext_hicma for more information.
 */
void tpqrt(Matrix& A, Matrix& B, Matrix& T);

} // namespace hicma

#endif // hicma_operations_LAPACK_h
