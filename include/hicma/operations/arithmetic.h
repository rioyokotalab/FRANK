#ifndef hicma_operations_arithmetic_h
#define hicma_operations_arithmetic_h

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"


namespace hicma
{

// Addition
Matrix& operator+=(Matrix&, const Matrix&);

Dense operator+(const Dense&, const Dense&);

// Subtraction
MatrixProxy operator-(const Matrix&, const Matrix&);

// Multiplication
Matrix& operator*=(Matrix&, double);

} // namespace hicma

#endif // hicma_operations_arithmetic_h
