#ifndef FRANK_util_l2_error_h
#define FRANK_util_l2_error_h


/**
 * @brief General namespace of the FRANK library
 */
namespace FRANK
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

} // namespace FRANK

#endif // FRANK_util_l2_error_h
