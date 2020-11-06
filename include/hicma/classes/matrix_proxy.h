#ifndef hicma_classes_matrix_proxy_h
#define hicma_classes_matrix_proxy_h

#include <memory>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

class MatrixProxy {
 private:
  std::unique_ptr<Matrix> ptr;
 public:
  // Special member functions
  MatrixProxy() = default;

  ~MatrixProxy() = default;

  MatrixProxy(const MatrixProxy& A);

  MatrixProxy& operator=(const MatrixProxy& A);

  MatrixProxy(MatrixProxy&& A) = default;

  MatrixProxy& operator=(MatrixProxy&& A) = default;

  // Additional constructors from Matrix to allow implicit conversion
  explicit MatrixProxy(const Matrix& A);

  MatrixProxy(Matrix&& A);

  // Conversion operator to Matrix&. We want to write our operations as
  // operation(Matrix&, Matrix&) and not have to write a list of overloads that
  // cover cases where we pass operation(H(0, 0), H(0, 1)).
  // If we define an implicit copy/move constructor on the Matrix class, the
  // derived types get cut short since we would need a copy/move to implement
  // them and would have no way of knowing which of the types we derived from
  // Matrix is actually pointed to by ptr of MatrixProxy.
  operator const Matrix&() const;

  operator Matrix&();
};

} // namespace hicma

#endif // hicma_classes_matrix_proxy_h
