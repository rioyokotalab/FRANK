#ifndef FRANK_operations_arithmetic_h
#define FRANK_operations_arithmetic_h

#include "FRANK/classes/dense.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/classes/matrix_proxy.h"


/**
 * @brief General namespace of the FRANK library
 */
namespace FRANK
{

/**
 * @brief Defines addition-assignment operator between two matrices
 *
 * @param A
 * `Matrix` instance
 * @param B
 * `Matrix` instance
 *
 * @return const Matrix&
 * Reference to the modified \p A that contains <tt>A+B</tt>
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_FRANK for more information.
 */
Matrix& operator+=(Matrix& A, const Matrix& B);

/**
 * @brief Defines addition operator between two `Dense` matrices
 *
 * @param A
 * `Dense` instance
 * @param B
 * `Dense` instance
 *
 * @return Dense
 * `Dense` matrix that contains <tt>A+B</tt>
 */
Dense operator+(const Dense&, const Dense&);

/**
 * @brief Defines subtraction operator between two matrices
 *
 * @param A
 * `Matrix` instance
 * @param B
 * `Matrix` instance
 *
 * @return MatrixProxy
 * Instance of `MatrixProxy` that owns a matrix containing <tt>A-B</tt>
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_FRANK for more information.
 */
MatrixProxy operator-(const Matrix& A, const Matrix& B);

/**
 * @brief Defines multiplication-assignment operator between a matrix and scalar value
 *
 * @param A
 * `Matrix` instance
 * @param b
 * Scalar value
 *
 * @return Matrix&
 * Reference to the modified \p A which contains <tt>A*b</tt>
 *
 * Definitions may differ depending on the types of the parameters.
 * Definition for each combination of types (subclasses of `Matrix`) is implemented as a specialization of \OMM.
 * The multi-dispatcher then will select the correct implementation based on the types of parameters given at runtime.
 * Read \ext_FRANK for more information.
 */
Matrix& operator*=(Matrix& A, const double b);

} // namespace FRANK

#endif // FRANK_operations_arithmetic_h
