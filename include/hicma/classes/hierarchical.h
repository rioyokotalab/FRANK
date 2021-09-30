/**
 * @file hierarchical.h
 * @brief Include the `Hierarchical` matrix class and some peripherals.
 *
 * @copyright Copyright (c) 2020
 */
#ifndef hicma_classes_hierarchical_h
#define hicma_classes_hierarchical_h

#include "hicma/definitions.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Dense;
class ClusterTree;
class MatrixInitializer;

/**
 * @brief Flexible class handling any kind of block-structured matrix.
 *
 * A `Hierarchical` matrix instance consists of a 2-dimensional array of
 * subblocks, each of which can have any type deriving from the base `Matrix`
 * class. Since the type of its submatrices generally depends on user input and
 * is thus unknown until runtime, the class treats its submatrices agnostic of
 * their type. The `MatrixProxy` class is central in achieving this.
 *
 * Although this class only has constructors and indexing member functions, it
 * the central piece of HiCMA - operations on this class are the main purpose of
 * the library.
 */
class Hierarchical : public Matrix {
 public:
  /**
   * @brief Dimension in terms of submatrices {block_rows, block_columns}
   */
  std::array<int64_t, 2> dim = {0, 0};
 private:
  std::vector<MatrixProxy> data;
 public:
  // Special member functions
  Hierarchical() = default;

  virtual ~Hierarchical() = default;

  Hierarchical(const Hierarchical& A) = default;

  Hierarchical& operator=(const Hierarchical& A) = default;

  Hierarchical(Hierarchical&& A) = default;

  Hierarchical& operator=(Hierarchical&& A) = default;

  /**
   * @brief Move from a `MatrixProxy` instance containing a `Hierarchical`
   * matrix
   *
   * @param A
   * `MatrixProxy` that must contain a `Hierarchical` instance.
   *
   * This move constructor is useful to move from a sub-matrix of another
   * `Hierarchical` instance that is known to be of `Hierarchical` type. Passing
   * a `MatrixProxy` to this constructor that does not contain a `Hierarchical`
   * instance will result in a runtime error.
   *
   * This constructor is intended to be a convenience method for testing code
   * and quick development. If you find yourself using this function often, you
   * should likely refactor your code to use an \OMM instead.
   */
  Hierarchical(MatrixProxy&& A);

  /**
   * @brief Construct a new `Hierarchical` object of given size
   *
   * @param n_row_blocks
   * Number of blocks rows of the new `Hierarchical` matrix.
   * @param n_col_blocks
   * Number of blocks columns of the new `Hierarchical` matrix.
   *
   * The new `Hierarchical` instance contains empty `MatrixProxy` subblocks.
   * Before using any submatrices, it is thus necessary to initialize them to
   * some child type of `Matrix`.
   */
  Hierarchical(int64_t n_row_blocks, int64_t n_col_blocks=1);

  /**
   * @brief General constructor of the `Hierarchical` class
   *
   * @param node
   * `ClusterTree` describing the desired structure of the `Hierarchical`
   * matrix.
   * @param initializer
   * Matrix element initializer. Several different initializers are available.
   *
   * This constructor is used by several other constructors that only directly
   * create a`ClusterTree` and some type of matrix element initializer. It
   * contains the recursion loop that will fill all subblocks of the
   * `Hierarchical` instance.
   */
  Hierarchical(
    const ClusterTree& node,
    MatrixInitializer& initializer
  );

  /**
   * @brief Construct a new `Hierarchical` matrix from a kernel and parameters
   *
   * @param kernel
   * Kernel used to compute matrix entries from together with \p params.
   * @param params
   * Vector with parameters used as input to the kernel.
   * @param n_rows
   * Number of rows of the new matrix.
   * @param n_cols
   * Number of columns of the new matrix.
   * @param rank
   * Fixed rank used for any `LowRank` approximations.
   * @param nleaf
   * Maximum size for leaf level submatrices.
   * @param admis
   * Admissibility in terms of distance from the diagonal of the matrix on the
   * current recursion level (for `POSITION_BASED_ADMIS`) or admissibility constant
   * (for `GEOMETRY_BASED_ADMIS`)
   * @param admis_type
   * Either `POSITION_BASED_ADMIS` or `GEOMETRY_BASED_ADMIS`
   * @param n_row_blocks
   * Number of blocks rows of the new `Hierarchical` matrix.
   * @param n_col_blocks
   * Number of blocks columns of the new `Hierarchical` matrix.
   * @param row_start
   * Starting index into the vector \p params of the rows of the new matrix.
   * @param col_start
   * Starting index into the vector \p params of the columns of the new matrix.
   *
   * The elements of the submatrices of the `Hierarchical` matrix will be
   * calculated according to the kernel function as well as the vector of values
   * passed to this function. In an application, the parameters could for
   * example be coordinates of particles, and the kernel could describe the
   * interaction of particles. The result would be an interaction matrix between
   * two groups of particles. Using \p row_start, \p col_start as well as \p
   * n_rows and \p n_cols, one can create this matrix for the interaction
   * between arbitrary sub-groups of the particles for which the parameters are
   * available in \p params.
   */
  Hierarchical(
    void (*kernel)(
      double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
      const std::vector<std::vector<double>>& params,
      int64_t row_start, int64_t col_start
    ),
    const std::vector<std::vector<double>>& params,
    int64_t n_rows, int64_t n_cols,
    int64_t rank,
    int64_t nleaf,
    double admis=0,
    int64_t n_row_blocks=2, int64_t n_col_blocks=2,
    int admis_type=POSITION_BASED_ADMIS,
    int64_t row_start=0, int64_t col_start=0
  );

  /**
   * @brief Construct a new `Hierarchical` matrix from a `Dense` matrix
   *
   * @param A
   * `Dense` matrix to be hierarchically compressed.
   * @param rank
   * Fixed rank used for any `LowRank` approximations.
   * @param nleaf
   * Maximum size for leaf level submatrices.
   * @param admis
   * Admissibility in terms of distance from the diagonal of the matrix on the
   * current recursion level.
   * @param n_row_blocks
   * Number of blocks rows of the new `Hierarchical` matrix.
   * @param n_col_blocks
   * Number of blocks columns of the new `Hierarchical` matrix.
   * @param basis_type
   * Type of basis according to #BasisType.
   * @param row_start
   * Starting index into the rows of \p A.
   * @param col_start
   * Starting index into the columns of \p A.
   *
   * If the input data is already available as a `Dense` matrix, use this
   * constructor to create the hierarchical compression.
   */
  Hierarchical(
    Dense&& A,
    int64_t rank,
    int64_t nleaf,
    double admis=0,
    int64_t n_row_blocks=2, int64_t n_col_blocks=2,
    int64_t row_start=0, int64_t col_start=0
  );

  Hierarchical(
    std::string filename, MatrixLayout ordering,
    const std::vector<std::vector<double>>& params,
    int64_t n_rows, int64_t n_cols,
    int64_t rank,
    int64_t nleaf,
    double admis=0,
    int64_t n_row_blocks=2, int64_t n_col_blocks=2,
    int admis_type=POSITION_BASED_ADMIS,
    int64_t row_start=0, int64_t col_start=0
  );

  /**
   * @brief Access elements of `Hierarchical` with a pair of indices
   *
   * @param pos
   * Pair of indices {block_row, block_col}.
   * @return const MatrixProxy&
   * Constant reference to the submatrix at index \p pos.
   *
   * Convenience function, mainly for indexing with ClusterTree::rel_pos.
   * The type of the return matrix contained the in the returned `MatrixProxy`
   * is generally runtime dependent.
   */
  const MatrixProxy& operator[](const std::array<int64_t, 2>& pos) const;

  /**
   * @brief Access elements of `Hierarchical` with a pair of indices
   *
   * @param pos
   * Pair of indices {block_row, block_col}.
   * @return const MatrixProxy&
   * Reference to the submatrix at index \p pos.
   *
   * Convenience function, mainly for indexing with ClusterTree::rel_pos.
   * The type of the return matrix contained the in the returned `MatrixProxy`
   * is generally runtime dependent.
   * It is possible to move it directly into a desired matrix type if the type
   * is known, for example directly after using ::split.
   */
  MatrixProxy& operator[](const std::array<int64_t, 2>& pos);

  // TODO Consider removing one-dimensional indexing.
  /**
   * @brief Access elements of `Hierarchical` matrix assuming it is vector-like
   *
   * @param i
   * Index into the vector-like hierarchical matrix.
   * @return const MatrixProxy&
   * Constant reference to the submatrix at index \p i.
   *
   * This indexing function should only be used when it is know then that
   * `Hierarchical` matrix is vector-like, that is if either `dim[0]==1` or
   * `dim[1]==1`. It works for both row and column vector-like matrices.
   * The type of the return matrix contained the in the returned `MatrixProxy`
   * is generally runtime dependent.
   */
  const MatrixProxy& operator[](int64_t i) const;

  /**
   * @brief Access elements of `Hierarchical` matrix assuming it is vector-like
   *
   * @param i
   * Index into the vector-like hierarchical matrix.
   * @return const MatrixProxy&
   * Reference to the submatrix at index \p i.
   *
   * This indexing function should only be used when it is know then that
   * `Hierarchical` matrix is vector-like, that is if either `dim[0]==1` or
   * `dim[1]==1`. It works for both row and column vector-like matrices.
   * The type of the return matrix contained the in the returned `MatrixProxy`
   * is generally runtime dependent.
   * It is possible to move it directly into a desired matrix type if the type
   * is known, for example directly after using ::split.
   */
  MatrixProxy& operator[](int64_t i);

  /**
   * @brief Access elements of `Hierarchical` with a row and column index
   *
   * @param i
   * Row block to be accessed.
   * @param j
   * Column block to be accessed.
   * @return const MatrixProxy&
   * Const reference to the submatrix at index {\p i, \p j}.
   *
   * The type of the return matrix contained the in the returned `MatrixProxy`
   * is generally runtime dependent.
   */
  const MatrixProxy& operator()(int64_t i, int64_t j) const;

  /**
   * @brief Access elements of `Hierarchical` with a row and column index
   *
   * @param i
   * Row block to be accessed.
   * @param j
   * Column block to be accessed.
   * @return const MatrixProxy&
   * Reference to the submatrix at index {\p i, \p j}.
   *
   * The type of the return matrix contained the in the returned `MatrixProxy`
   * is generally runtime dependent.
   * It is possible to move it directly into a desired matrix type if the type
   * is known, for example directly after using ::split.
   */
  MatrixProxy& operator()(int64_t i, int64_t j);
};

} // namespace hicma

#endif // hicma_classes_hierarchical_h
