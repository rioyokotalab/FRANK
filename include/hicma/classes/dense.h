#ifndef hicma_classes_dense_h
#define hicma_classes_dense_h

#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>


namespace hicma
{

class DataHandler;
class IndexRange;
class Task;


/**
 * @brief Class handling a regular dense matrix
 *
 * In HiCMA, only the `Dense` class handles actual matrix elements. All other
 * classes are composites of `Dense` matrices.
 */
class Dense : public Matrix {
  // TODO Find way to avoid using friend here! Best not to rely on it.
  // Also don't wanna expose the DataHandler directly though...
  friend Task;
 public:
  /**
   * @brief Dimension of the matrix {rows, columns}
   */
  std::array<int64_t, 2> dim = {0, 0};
  /**
   * @brief Stride of the array in memory
   *
   * If the matrix is stored as an continuous block, then `stride = dim[1]`
   * since row-major storage is used. If the matrix handled by this instance of
   * `Dense` is a sub-matrix of a larger matrix, then `stride` can be larger
   * than `dim[1]`.
   */
  int64_t stride = 0;
 private:
  // Handler of the representation in memory
  std::shared_ptr<DataHandler> data;
  // Relative position inside a possible larger array in memory.
  std::array<int64_t, 2> rel_start = {0, 0};
  // Pointer used to speed up the indexing into submatrices. Will point to the
  // beginning of the array in memory. Without this pointer, rel_start would
  // need to be used every time the indexing operator is called, leading to
  // measurable performance decrease.
  double* data_ptr = nullptr;
 public:
  Dense() = default;

  virtual ~Dense() = default;

  /**
   * @brief Copy constructor
   *
   * @param A
   * `Dense` object to be copied.
   *
   * This will create a deep copy of the `Dense` object that passed to this
   * constructor.
   */
  Dense(const Dense& A);

  /**
   * @brief Copy assignment operator
   *
   * @param A
   * `Dense` object to be copied.
   *
   * @return Dense&
   * Reference to the modified `Dense` object.
   *
   * Make the left hand side of the operator a deep copy of the right hand side.
   */
  Dense& operator=(const Dense& A);

  Dense(Dense&& A) = default;

  Dense& operator=(Dense&& A) = default;

  /**
   * @brief Explicit copy/conversion from any `Matrix` type using an \OMM
   *
   * @param A
   * `Matrix` object from which a `Dense` matrix is to be generated.
   *
   * Any class derived from `Matrix` can be passed and a new `Dense` matrix of
   * the same size will be generated from it. If a `Dense` matrix is passed,
   * this equivalent to the copy constructor `Dense(const Dense&)`. In the case
   * of a `LowRank` matrix, the factors `LowRank::U`, `LowRank::S` and
   * `LowRank::V` will be multiplied together to create the new matrix `D`
   * (\f$D=U*S*V\f$).
   *
   * An \OMM is used to select the correct implementation. For new types, a new
   * specialization of this \OMM will need to be implemented. Read \ext_hicma
   * for more information.
   *
   * Note that this operator is set to `explicit` so as to avoid inadvertently
   * converting matrices to `Dense` matrices, as doing so would likely cause
   * drastic performance drops.
   */
  explicit Dense(const Matrix& A);

  /**
   * @brief Move from a `MatrixProxy` instance
   *
   * @param A
   * `MatrixProxy` that must contain a `Dense` instance.
   *
   * This move constructor is useful to move from a sub-matrix of a
   * `Hierarchical` instance that is known to be of `Dense` type. Passing a
   * `MatrixProxy` to this constructor that does not contain a `Dense` instance
   * will result in a runtime error.
   *
   * This constructor is intended to be a convenience method for testing code
   * and quick development. If you find yourself using this function often, you
   * should likely refactor your code to use an \OMM instead.
   */
  Dense(MatrixProxy&& A);

  /**
   * @brief Construct a new `Dense` object given the desired size
   *
   * @param n_rows
   * Desired number of rows of the created matrix.
   * @param n_cols
   * Desired number of column of the created matrix.
   *
   * All elements of the newly created `Dense` matrix will be initialized to 0.
   */
  Dense(int64_t n_rows, int64_t n_cols=1);

  // TODO Add overload where vector doesn't need to be passed. That function
  // should forward to this one with a 0-sized vector. This is to make
  // initialization with functions like identity and random_uniform easier.
  /**
   * @brief Construct a new `Dense` object from a kernel function
   *
   * @param kernel
   * Kernel used to compute matrix entries from the
   * @param params
   * Vector with parameters used as input to the kernel
   * @param n_rows
   * Number of rows of the new matrix.
   * @param n_cols
   * Number of columns of the new matrix.
   * @param row_start
   * Starting index into the vector params of the rows of the new matrix.
   * @param col_start
   * Starting index into the vector params of the columns of the new matrix.
   *
   * The elements of the new `Dense` matrix will be calculated according to the
   * kernel function as well as the vector of values passed to this function.
   * In an application, the parameters could for example be coordinates of
   * particles, and the kernel could describe the interaction of particles.
   * The result would be an interaction matrix between two groups of particles.
   * Using \p row_start, \p col_start as well as \p n_rows and \p n_cols, one
   * can create this matrix for the interaction between arbitrary sub-groups of
   * the particles for which the parameters are available in \p params. This can
   * for example be used to generate the `Dense` blocks of a `Hierarchical`
   * matrix.
   */
  Dense(
    void (*kernel)(
      double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
      const std::vector<std::vector<double>>& params,
      int64_t row_start, int64_t col_start
    ),
    const std::vector<std::vector<double>>& params,
    int64_t n_rows, int64_t n_cols=1,
    int64_t row_start=0, int64_t col_start=0
  );

  /**
   * @brief Assign the constant value \p a to all elements of the `Dense` matrix
   *
   * @param a
   * Value to assign to all elements of the `Dense` matrix.
   * @return const Dense&
   * Reference to the modified `Dense` matrix.
   */
  Dense& operator=(const double a);

  // TODO Consider removing one-dimensional indexing.
  /**
   * @brief Access elements of `Dense` matrix assuming it is a vector
   *
   * @param i
   * Index into the vector
   * @return double&
   * Reference to the vector element at index \p i
   *
   * This indexing function should only be used when it is know then that
   * `Dense` matrix is a vector, that is if either `dim[0]==1` or `dim[1]==1`.
   * It works for both row and column vectors, and also if the vector is part of
   * a larger matrix in memory (stride is used).
   */
  double& operator[](int64_t i);

  /**
   * @brief Access elements of `Dense` matrix assuming it is a vector
   *
   * @param i
   * Index into the vector
   * @return double&
   * Constant reference to the vector element at index \p i
   *
   * Same as `operator[](int64_t)`, but a constant reference is returned.
   */
  const double& operator[](int64_t i) const;

  /**
   * @brief Access elements of `Dense` matrix
   *
   * @param i
   * Row index into the matrix
   * @param j
   * Column index into the matrix
   * @return double&
   * Reference to the matrix element at (\p i, \p j)
   */
  double& operator()(int64_t i, int64_t j);

  /**
   * @brief Access elements of `Dense` matrix
   *
   * @param i
   * Row index into the matrix
   * @param j
   * Column index into the matrix
   * @return double&
   * Constant reference to the matrix element at (\p i, \p j)
   */
  const double& operator()(int64_t i, int64_t j) const;

  /**
   * @brief Get pointer to start of array in memory
   *
   * @return double*
   * Pointer to the beginning of the continuous array in memory.
   *
   * Many functions, such as BLAS/LAPACK routines, require passing pointers to
   * continuous arrays in memory. This operator returns the pointer for a
   * `Dense` matrix.
   */
  double* operator&();


  /**
   * @brief Get pointer to start of array in memory
   *
   * @return double*
   * Pointer to the beginning of the constant continuous array in memory.
   *
   * Same as `::operator&()`, but returns a constant pointer.
   */
  const double* operator&() const;

  // Utility methods
  /**
   * @brief Create a shallow copy with a shared memory representation
   *
   * @return Dense
   * Shallow copy of the matrix this operator is called on.
   *
   * The created shallow copy and this `Dense` instance share ownership of the
   * representation in memory. This means the memory will remain allocated until
   * all shared `Dense` instances are deleted.
   */
  Dense shallow_copy() const;

  /**
   * @brief Check if this `Dense` instance is shared with another instance
   *
   * @param A
   * `Dense` instance to check sharedness for
   * @return true
   * if this `Dense` matrix is shared with the instance \p A
   * @return false
   * if the matrices are not shared
   *
   * Shared here means that they manage the exact same memory, so that
   * modifications to one will also modify the other.
   */
  bool is_shared_with(const Dense& A) const;

  /**
   * @brief Check if this `Dense` instance is a submatrix of a larger matrix.
   *
   * @return true if this `Dense` instance is a part of a larger matrix
   * @return false if it has no `Dense` parent matrix
   *
   * Note that this does not check if the `Dense` instance is part of a
   * `Hierarchical` matrix. This check is only concerned with whether it covers
   * the entirety of the floating point array in memory, or whether it only
   * covers a smaller part.
   */
  bool is_submatrix() const;

  /**
   * @brief Split the matrix according to row and column index ranges
   *
   * @param row_ranges
   * Set of non-overlapping ranges whose union is the full row index range of
   * this `Dense` instance.
   * @param col_ranges
   * Set of non-overlapping ranges whose union is the full column index range of
   * this `Dense` instance.
   * @param copy
   * If set to true, the matrices will be deep copies. Otherwise, they the split
   * will be performed without any copies, and the parent and child matrices
   * will be dependent.
   * @return std::vector<Dense>
   * Vector containing the resulting submatrices.
   *
   * If there are both multiple row and column ranges, then the one-dimensional
   * vector returned represents a two-dimensional structure. Just like with the
   * two-dimensional floating point arrays in memory, we use row-major storage.
   */
  std::vector<Dense> split(
    const std::vector<IndexRange>& row_ranges,
    const std::vector<IndexRange>& col_ranges,
    bool copy=false
  ) const;

  /**
   * @brief Split the matrix into a number of parts along rows and columns
   *
   * @param n_row_splits
   * Number of splits along the rows of the matrix.
   * @param n_col_splits
   * Number of splits along the columns of the matrix.
   * @param copy
   * If set to true, the matrices will be deep copies. Otherwise, they the split
   * will be performed without any copies, and the parent and child matrices
   * will be dependent.
   * @return std::vector<Dense>
   * Vector containing the resulting submatrices.
   *
   * If there are both multiple row and column ranges, then the one-dimensional
   * vector returned represents a two-dimensional structure. Just like with the
   * two-dimensional floating point arrays in memory, we use row-major storage.
   *
   * If the number of rows of the matrix is not divisible by \p n_row_splits,
   * then the last split will be smaller than the other splits. The same is true
   * for the column splits.
   */
  std::vector<Dense> split(
    uint64_t n_row_splits, uint64_t n_col_splits, bool copy=false
  ) const;
};

} // namespace hicma

#endif // hicma_classes_dense_h
