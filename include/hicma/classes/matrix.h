#ifndef hicma_classes_matrix_h
#define hicma_classes_matrix_h


namespace hicma
{

class Matrix {
 public:
  // Special member functions
  Matrix() = default;

  virtual ~Matrix() = default;

  Matrix(const Matrix& A) = default;

  Matrix& operator=(const Matrix& A) = default;

  Matrix(Matrix&& A) = default;

  Matrix& operator=(Matrix&& A) = default;
};

} // namespace hicma

#endif // hicma_classes_matrix_h
