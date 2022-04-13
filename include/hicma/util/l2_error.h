#ifndef hicma_util_l2_error_h
#define hicma_util_l2_error_h


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

/**
 * @brief Calculate the Frobenius norm of the difference between two `Matrices`
 * 
 * Square root of the sum of squares of the entry-wise differnce
 * between two `Matrices`.
 * 
 * @param A input matrix
 * @param B input matrix
 * @return double Frobenius norm of A-B
 */
double l2_error(const Matrix& A, const Matrix& B);

} // namespace hicma

#endif // hicma_util_l2_error_h
