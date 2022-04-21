#ifndef FRANK_operations_LAPACK_h
#define FRANK_operations_LAPACK_h

#include <cstdint>
#include <tuple>
#include <vector>


/**
 * @brief General namespace of the FRANK library
 */
namespace FRANK
{

class Matrix;
class MatrixProxy;
class Dense;
class Hierarchical;

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
 * Read \ext_FRANK for more information.
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
 * @return
 * Tuple containing two values: the matrix \p V and chosen column indices of \p A
 *
 * This method performs one-sided interpolative decomposition of a given matrix.
 * The matrix \p A is approximated as
 *
 * <tt>A &asymp; U * V</tt>
 *
 * Where <tt>U</tt> is a M-by-k matrix composed of the chosen \p k columns of \p A and
 * <tt>V</tt> is a k-by-N matrix.
 */
std::tuple<Dense, std::vector<int64_t>> one_sided_id(Matrix& A, const int64_t k);

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
  const int mode,
  const double cond,
  const double dmax,
  const int kl, const int ku,
  const char& pack,
  Dense& A
);

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

/**
 * @brief Compute the singular values of a `Dense` matrix
 *
 * @param A
 * M-by-N `Dense` matrix
 *
 * @return a vector containing singular values of \p A
 *
 * This method uses Singular Value Decomposition to compute the singular values of \p A.
 * The obtained singular values are sorted descendingly.
 */
std::vector<double> get_singular_values(Dense& A);

/**
 * @brief Compute two-sided interpolative decomposition (ID) of a `Dense` matrix
 *
 * @param A
 * M-by-N `Dense` instance to be factorized. Modified on finish.
 * @param k
 * Determines how many columns that needs to be taken (stopping criterion)
 *
 * @return
 * Tuple containing three matrices: `U`, `S`, and `V`.
 *
 * This method performs two-sided interpolative decomposition of a given matrix.
 * The matrix \p A is approximated as
 *
 * <tt>A &asymp; U * S * V</tt>
 *
 * Where <tt>U</tt> is a M-by-k matrix, <tt>S</tt> is a k-by-k matrix, and <tt>V</tt> is a k-by-N matrix.
 */
std::tuple<Dense, Dense, Dense> id(Matrix& A, const int64_t k);

/**
 * @brief Compute Householder QR factorization with column pivoting
 *
 * @param A
 * M-by-N `Dense` instance to be factorized. Overwritten on finish
 *
 * @return
 * Tuple containing the upper triangular matrix <tt>R</tt> and permuted column indices
 *
 * This method performs Householder QR factorization with column pivoting. On finish, the lower trapezoidal part of \p A will be overwritten with householder reflectors representing the orthogonal factor Q.
 */
std::tuple<Dense, std::vector<int64_t>> geqp3(Matrix& A);

/**
 * @brief Compute truncated Householder QR factorization with column pivoting
 *
 * @param A
 * M-by-N `Dense` instance to be factorized
 * @param eps
 * Relative error threshold
 *
 * @return
 * Tuple containing the matrices <tt>Q</tt> and <tt>R</tt>.
 *
 * Truncation is performed based on the specified relative error threshold.
 * The matrix \p A is approximated as
 *
 * <tt>A &asymp; Q * R</tt>
 *
 * such that <tt>Q</tt> is a M-by-k matrix with orthonormal columns, <tt>R</tt> is a k-by-N matrix,
 * and k is the minimum rank that satisfies
 *
 * <tt>|A-QR|_F &le; eps * |A|_F</tt>,
 *
 * where |A|_F denotes the Frobenius norm of \p A.
 */
std::tuple<Dense, Dense> truncated_geqp3(const Dense& A, const double eps);

/**
 * @brief Orthogonalize a block column of a BLR or H-matrix
 *
 * @param j
 * Block column index
 * @param A
 * `Hierarchical` object representing BLR or H-matrix to be orthogonalized
 * @param Q
 * `Hierarchical` object with same dimension as \p A to store orthogonalized block column
 * @param R
 * `Matrix` object to store the resulting upper triangular factor
 *
 * This function performs orthogonalization of a block column using Modified Gram Schmidt iteration based on
 * the method presented in (Peter Benner and Thomas Mach, 2010).
 * On finish, the orthogonalized block column will overwrite the <tt>j</tt>-th block column of \p Q.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_FRANK for more information.
 */
void orthogonalize_block_col(const int64_t j, const Matrix& A, Matrix& Q, Matrix& R);

/**
 * @brief Compute reduced QR factorization of a general matrix using Modified Gram-Schmidt iteration
 *
 * @param A
 * M-by-N `Matrix` to be factorized. Overwritten with \p Q on finish
 * @param R
 * N-by-N `Matrix` that is overwritten with upper triangular factor \p R on finish
 *
 * This function performs MGS QR factorization
 * Prior to calling, it is assumed that \p R have been initialized with proper dimension.
 * Upon finish, \p A will be overwritten with the resulting orthogonal factor \p Q.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_FRANK for more information.
 */
void mgs_qr(Matrix& A, Matrix& Q, Matrix& R);

/**
 * @brief Compute QR factorization of a general matrix using Householder method
 *
 * @param A
 * M-by-N `Matrix` to be factorized. Overwritten on finish
 * @param Q
 * M-by-N `Matrix` with the same type as \p A. Overwritten on finish with orthogonal factor on finish
 * @param R
 * N-by-N `Matrix` with the same type as \p A. Overwritten on finish with upper triangular factor
 *
 * This method perform Householder QR factorization of a given matrix.
 * Prior to calling, \p Q and \p R need to be initialized with proper dimensions.
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_FRANK for more information.
 */
void qr(Matrix& A, Matrix& Q, Matrix& R);

/**
 * @brief Triangularize a block column of a BLR-matrix
 *
 * @param j
 * Block column index
 * @param A
 * `Hierarchical` object representing the BLR-matrix to be triangularized
 * @param T
 * `Hierarchical` object to store intermediate T matrices of Compact WY Representation
 *
 * This function performs triangularization of a block column using Householder reflection based on
 * the method presented in (Apriansyah and Yokota, 202X).
 * On finish, the <tt>j</tt>-th block column of \p A will be overwritten with the resulting upper triangular factor and unit lower trapezoidal \p Y_j matrix.
 * The orthogonal factor \p Q is stored in Compact WY representation:
 *
 * <tt>Q_j = I - (Y_j * T_j * transpose(Y_j))</tt>
 */
void triangularize_block_col(const int64_t j, Hierarchical& A, Hierarchical& T);

/**
 * @brief Apply an orthogonal factor from FRANK::triangularize_block_col to a block column of a BLR-matrix
 *
 * @param Y
 * @param T
 * @param k
 * Block column index of \p Y (and \p T) that implicitly stores the orthogonal factor
 * @param trans
 * \p true if \p transpose(Q_k) will be used, \p false otherwise
 * @param A
 * BLR-matrix whose <tt>j</tt>-th block column will be multiplied with orthogonal factor
 * @param j
 * Block column index of \p A
 *
 * This function performs the following operation
 *
 * <tt>A[*,j] = Q_k * A[*,j]</tt> &nbsp;&nbsp;or &nbsp;&nbsp;<tt>A[*,j] = transpose(Q_k) * A[*,j]</tt>
 *
 * where \p Q_k is the orthogonal factor coming from FRANK::triangularize_block_col stored in compact WY representation:
 *
 * <tt>Q_k = I - (Y[*,k] * T[k] * transpose(Y[*,k]))
 */
void apply_block_col_householder(
    const Hierarchical& Y, const Hierarchical& T, const int64_t k, const bool trans,
    Hierarchical& A, const int64_t j
);

/**
 * @brief Perform Blocked Householder BLR-QR Factorization
 *
 * @param A
 * Block Low-Rank matrix with p-by-q blocks. Overwritten on finish
 * @param T
 * Block Dense matrix with q-by-1 blocks. Overwritten on finish
 *
 * This method computes the QR factorization of a Block Low-Rank matrix \p A using blocked Householder technique.
 * Prior to calling, \p T needs to be initialized with proper dimension.
 * The BLR matrix \p A is factorized as
 *
 * <tt>A &asymp; Q * R</tt>,
 *
 * where \p Q is an orthogonal BLR matrix with p-by-p blocks and \p R is an upper triangular BLR matrix with p-by-q blocks.
 * On finish, \p R is stored in the upper triangular part of \p A.
 * \p Q is implicitly stored as \p Y and \p T such that \p Y contains Householder vectors stored in the strictly lower trapezoidal part of \p A and each block in \p T is an upper triangular matrix.
 */
void blocked_householder_blr_qr(Hierarchical& A, Hierarchical& T);

/**
 * @brief Perform left multiplication by Q from blocked Householder BLR-QR
 *
 * @param Y
 * Block Low-Rank matrix with p-by-q blocks containing Householder vectors in its strictly lower trapezoidal part
 * @param T
 * Block Dense matrix with q-by-1 blocks
 * @param C
 * Block Low-Rank matrix C with p-by-q blocks
 * @param trans
 * \p true if \p transpose(Q) will be used, \p false otherwise
 *
 * This method performs the following operation
 *
 * <tt>C = Q*C</tt> or <tt>C = transpose(Q)*C</tt>
 *
 * where \p Q is the orthogonal p-by-p blocks BLR matrix coming from the blocked Householder BLR-QR factorization that is stored in \p Y and \p T matrices.
 */
void left_multiply_blocked_reflector(
    const Hierarchical& Y, const Hierarchical& T, Hierarchical& C, const bool trans
);

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
 * Read \ext_FRANK for more information.
 *
 * <b>Note</b> that for any type of \p A, at the lowest level it will end up with operations involving only `Dense` matrices.
 * Thus at the core, this method relies on <a target="_blank" href="http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga3ad112f2b0890b3815e696628906f30c.html#ga3ad112f2b0890b3815e696628906f30c"><tt>dgeqrt3</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation for more information.
 */
void geqrt(Matrix& A, Matrix& T);

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
 * Read \ext_FRANK for more information.
 *
 * <b>Only support \p V and \p T of type `Dense` at the moment</b>. Also <b>note</b> that for any combination of types, at the lowest level it will end up with operations involving only `Dense` matrices.
 * Thus at the core, this method relies on <a target="_blank" href="http://www.netlib.org/lapack/explore-html/d8/d9b/group__double_o_t_h_e_rauxiliary_ga83c81583bd444e0cf021fb006cd9a5e8.html#ga83c81583bd444e0cf021fb006cd9a5e8"><tt>dlarfb</tt></a> subroutine provided by BLAS/LAPACK.
 * See the documentation for more information.
 */
void larfb(const Matrix& V, const Matrix& T, Matrix& C, const bool trans);

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
 * Read \ext_FRANK for more information.
 */
void tpqrt(Matrix& A, Matrix& B, Matrix& T);

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
 * Read \ext_FRANK for more information.
 */
void tpmqrt(const Matrix& V, const Matrix& T, Matrix& A, Matrix& B, const bool trans);

/**
 * @brief Perform Tiled Householder BLR-QR Factorization
 *
 * @param A
 * Block Low-Rank matrix with p-by-q blocks. Overwritten on finish
 * @param T
 * Block Dense matrix with p-by-q blocks. Overwritten on finish
 *
 * This method computes the QR factorization of a Block Low-Rank matrix \p A using tiled Householder technique.
 * Prior to calling, \p T needs to be initialized with proper dimension
 * The BLR matrix \p A is factorized as
 *
 * <tt>A &asymp; Q * R</tt>,
 *
 * where \p Q is an orthogonal BLR matrix with p-by-p blocks and \p R is an upper triangular BLR matrix with p-by-q blocks.
 * On finish, \p R is stored in the upper triangular part of \p A.
 * \p Q is implicitly stored as \p Y and \p T such that \p Y contains Householder vectors stored in the strictly lower trapezoidal part of \p A and each block in \p T is an upper triangular matrix.
 */
void tiled_householder_blr_qr(Hierarchical& A, Hierarchical& T);

/**
 * @brief Perform left multiplication by Q from tiled Householder BLR-QR
 *
 * @param Y
 * Block Low-Rank matrix with p-by-q blocks containing Householder vectors in its strictly lower trapezoidal part
 * @param T
 * Block Dense matrix with p-by-q blocks
 * @param C
 * Block Low-Rank matrix C with p-by-q blocks
 * @param trans
 * \p true if \p transpose(Q) will be used, \p false otherwise
 *
 * This method performs the following operation
 *
 * <tt>C = Q*C</tt> or <tt>C = transpose(Q)*C</tt>
 *
 * where \p Q is the orthogonal p-by-p blocks BLR matrix coming from the tiled Householder BLR-QR factorization that is stored in \p Y and \p T matrices.
 */
void left_multiply_tiled_reflector(
    const Hierarchical& Y, const Hierarchical& T, Hierarchical& C, const bool trans
);

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

} // namespace FRANK

#endif // FRANK_operations_LAPACK_h
