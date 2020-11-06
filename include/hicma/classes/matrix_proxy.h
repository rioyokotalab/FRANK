#ifndef hicma_classes_matrix_proxy_h
#define hicma_classes_matrix_proxy_h

#include <memory>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

/**
 * @brief Proxy class that can contain any type derived from `Matrix`
 *
 * It is often necessary to return an arbitrary matrix type from a function, or
 * in the case of Hierarchical::operator() some object that any type of matrix
 * can be assigned to. Returning a `Matrix&` is insufficient for both cases,
 * thus this class is necessary. It owns (in terms of owning the instance of a
 * C++ class when it comes to memory management) an instance of any child class
 * of `Matrix`. When assigned to, it can replace the instance it holds with any
 * other instance through copy or move operations, and can act as a `Matrix&`
 * through a conversion operator.
 */
class MatrixProxy {
 private:
  std::unique_ptr<Matrix> ptr;
 public:
  // Special member functions
  MatrixProxy() = default;

  ~MatrixProxy() = default;

  /**
   * @brief Copy constructor
   *
   * @param A
   * MatrixProxy to be copied.
   *
   * Constructs a deep copy of whatever matrix type is contained in \p A. This
   * is done via an \OMM.
   */
  MatrixProxy(const MatrixProxy& A);

  /**
   * @brief Copy assignment operator
   *
   * @param A
   * MatrixProxy to be copied.
   * @return MatrixProxy&
   * Reference to the modified `MatriProxy` instance.
   *
   * Assigns a deep copy of whatever matrix type is contained in \p A to the
   * instance this is called on.
   */
  MatrixProxy& operator=(const MatrixProxy& A);

  MatrixProxy(MatrixProxy&& A) = default;

  MatrixProxy& operator=(MatrixProxy&& A) = default;

  /**
   * @brief Conversion copy constructor from any matrix type
   *
   * @param A
   * Reference to an instance of any type derived from `Matrix`.
   *
   * Acts the same way as `MatrixProxy(const MatrixProxy&)`, using an \OMM to
   * acquire the runtime type of \p A.
   */
  explicit MatrixProxy(const Matrix& A);

  /**
   * @brief Conversion move constructor from any matrix type
   *
   * @param A
   * Rvalue reference to an instance of any type derived from `Matrix`.
   *
   * Acts similar to `MatrixProxy(const Matrix&)`, using an \OMM to acquire the
   * runtime type of \p A. However, a move operation is used instead of a deep
   * copy, meaning that this constructor is generally much faster.
   */
  MatrixProxy(Matrix&& A);

  /**
   * @brief Conversion operator to a `Matrix&`
   *
   * @return const Matrix&
   * Reference to the `Matrix` contained in this `MatrixProxy`
   *
   * Generally, operations are defined on `Matrix&` and not `MatrixProxy&` (for
   * reasons of clarity and avoidance of code duplication). Conversion operators
   * are somewhat arcane C++ idioms, but in this case the only way to obtain a
   * reference of `Matrix&` type since the type contained in this `MatrixProxy`
   * is unknown at compile time. Users generally don't need to use this
   * conversion explicitly, as it will be performed automatically when passing a
   * `MatrixProxy` to any function accepting a `Matrix&`.
   */
  operator Matrix&();

  // If we define an implicit copy/move constructor on the Matrix class, the
  // derived types get cut short since we would need a copy/move to implement
  // them and would have no way of knowing which of the types we derived from
  // Matrix is actually pointed to by ptr of MatrixProxy.
  /**
   * @brief Conversion operator to a `const Matrix&`
   *
   * @return const Matrix&
   * Reference to the `Matrix` contained in this `MatrixProxy`
   *
   * Same as \ref operator Matrix&(), but returning a constant reference.
   */
  operator const Matrix&() const;
};

} // namespace hicma

#endif // hicma_classes_matrix_proxy_h
